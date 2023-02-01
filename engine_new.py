import time

import torch.optim as optim
from model import *
import util
import math

import torch.nn as nn
import torch.distributions as Dist

from tensorboardX import SummaryWriter
import wandb


def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask
    return hook


class FixedResCov(nn.Module):
    def __init__(self, n_components, n_vars, num_pred, rho=0.5, diag=False, trainL=True, device='cpu'):
        super(FixedResCov, self).__init__()
        self.dim_L_1 = (n_components, n_vars, n_vars)
        self.dim_L_2 = (n_components, num_pred, num_pred)

        init_L1 = torch.diag_embed(torch.rand(*self.dim_L_1[:2])) * 0.01
        init_L2 = torch.diag_embed(torch.rand(*self.dim_L_2[:2])) * 0.01

        self._L1 = nn.Parameter(init_L1.detach(), requires_grad=trainL)
        self._L2 = nn.Parameter(init_L2.detach(), requires_grad=trainL)
        self.diag = diag
        self.rho = rho

    @property
    def L1(self):
        # Ltemp = torch.tanh(self._L) * self.rho
        Ltemp = self._L1
        # Ltemp = Ltemp / Ltemp[:, 0, 0].unsqueeze(-1).unsqueeze(-1)

        if self.diag:
            return torch.diag_embed(torch.diagonal(Ltemp, dim1=1, dim2=2))
        else:
            return torch.tril(Ltemp)

    @property
    def L2(self):
        Ltemp = self._L2
        # Ltemp = Ltemp / Ltemp[:, 0, 0].unsqueeze(-1).unsqueeze(-1)

        if self.diag:
            return torch.diag_embed(torch.diagonal(Ltemp, dim1=1, dim2=2))
        else:
            return torch.tril(Ltemp)


class CholeskyResHead(nn.Module):
    def __init__(self, n_vars, n_components, pred_len=12, rho=0.1, loss="mse"):
        super(CholeskyResHead, self).__init__()
        self.n_vars = n_vars
        self.n_rank = n_components
        self.pred_len = pred_len

        self.rho = rho
        self.training = True

        self.loss = loss

    def forward(self, features):
        target = features['target']

        unscaled_target = features["unscaled_target"]

        mask = (unscaled_target != 0)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        # initialize nll_loss and mse_loss as torch tensor
        nll_loss = torch.tensor(10000)
        mse_loss = torch.tensor(100)

        if not self.rho == 0:
            nll_loss = self.get_nll(features, target, mask).mean()

       # if not self.rho == 1:
        predict = (features["mu"] * features['w'].exp()[..., 0].unsqueeze(1).unsqueeze(1)).sum(-1)

        target = features["target"]
        unscaled_target = features["unscaled_target"]

        mask = (unscaled_target != 0)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        if self.loss == "mae":
            mse_loss = torch.abs(predict-target)
        elif self.loss == "mse":
            mse_loss = (predict-target) ** 2
        else:
            raise NotImplementedError

        mse_loss = mse_loss * mask

        mse_loss = torch.where(torch.isnan(mse_loss), torch.zeros_like(mse_loss), mse_loss)

        mse_loss = torch.mean(mse_loss)

        loss = self.rho * nll_loss + (1-self.rho) * mse_loss
        return loss, nll_loss.item(), mse_loss.item()

    def get_nll(self, features, target, mask):
        mu, L_s, L_t = self.get_parameters(features)
        # w = features["w"]
        logw = features["w"].squeeze(-1)

        b, n, t, r = mu.shape
        n = self.n_vars
        t = len(self.pred_len)

        R_ext = (mu - target.unsqueeze(-1))
        R_flatten = R_ext.permute(0, 3, 1, 2)

        L_t = L_t.unsqueeze(0).repeat(b, 1, 1, 1)  # L_t L_tT = prc_T
        L_s = L_s.unsqueeze(0).repeat(b, 1, 1, 1)  # L_s L_sT = prc_S

        Ulogdet = L_s.diagonal(dim1=-1, dim2=-2).log().sum(-1)
        Vlogdet = L_t.diagonal(dim1=-1, dim2=-2).log().sum(-1)

        Q_t = torch.einsum("brij,brjk,brkl->bril", L_s.transpose(-1, -2), R_flatten, L_t)
        mahabolis = -0.5 * torch.pow(Q_t, 2).sum((-1, -2))

        nll = -n*t/2 * math.log(2*math.pi) + mahabolis + n * Vlogdet + t * Ulogdet + logw

        nll = - torch.logsumexp(nll, dim=1)
        # print(f"maxnll {nll.max():.5f} minnll {nll.min():.5f}")
        return nll

    def sample(self, features, nsample=None):
        mu, R, L_s, L_t = self.get_parameters(features)
        b, nt, r = R.shape
        n = self.n_vars
        t = len(self.pred_len)

        U = torch.inverse(L_s).unsqueeze(0).repeat(b, 1, 1, 1)
        V = torch.inverse(L_t).unsqueeze(0).repeat(b, 1, 1, 1)
        mu = mu.reshape(b, n, t)

        device = mu.device
        iid_dist = Dist.Independent(Dist.Normal(torch.zeros((b, r+1, n, t), device=device), torch.ones((b, r+1, n, t), device=device)), 1)

        if n is None:
            samples = iid_dist.sample()
        else:
            samples = iid_dist.sample((nsample,))

        samples = mu.unsqueeze(0) + torch.einsum("brln,sbrnt, brtk->sbrlk", U, samples, V).sum(2)

        return samples
        # torch.einsum("ln,nt,tk->lk", U[0],samples[0,0],V[0])

    def get_parameters(self, features):
        return features['mu'], features["L_spatial"], features["L_temporal"]


class MDN_trainer():
    def __init__(self, scaler, args, device, supports, aptinit, summary=True):

        self.args = args

        self.num_nodes = args.num_nodes
        self.n_components = args.n_components
        self.nhid = args.nhid
        self.pred_len = args.pred_len
        self.diag = args.diag
        self.rho = args.rho
        self.loss = args.loss
        self.scaler = scaler
        self.mix_mean = args.mix_mean

        self.device = device

        # self.out_per_comp = 2
        self.num_pred = len(self.pred_len) if isinstance(self.pred_len, list) else 1

        self.out_per_comp = self.n_components * self.num_pred

        self.model = gwnet(device, self.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=aptinit,
                           in_dim=args.in_dim, out_dim=self.out_per_comp, residual_channels=self.nhid, dilation_channels=self.nhid, skip_channels=self.nhid * 8, end_channels=self.nhid * 16)

        self.res_head = CholeskyResHead(self.num_nodes, self.n_components, pred_len=self.pred_len, rho=self.rho, loss=self.loss)

        self.covariance = FixedResCov(self.n_components, self.num_nodes, self.num_pred, rho=self.rho,
                                      diag=self.diag, trainL=self.rho != 0)

        self.fc_w = nn.Sequential(
            nn.Linear(self.num_nodes * self.num_pred, self.nhid),
            nn.ReLU(),
            nn.Linear(self.nhid, self.nhid),
            nn.ReLU(),
            nn.Linear(self.nhid, 1)
        )

        self.model_list = nn.ModuleDict(
            {
                'model': self.model,
                'res_head': self.res_head,
                'covariance': self.covariance,
                'fc_w': self.fc_w
            }
        )
        self.model_list.to(device)

        wandb.watch(self.model_list)
        # torch.autograd.set_detect_anomaly(True)
        self.optimizer = optim.Adam(self.model_list.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.clip = 5

        import datetime
        self.logdir = f'./logs/GWN_DynMix_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}{self.loss}loss_N{self.n_components}_nhid{self.nhid}_pred{self.num_pred}_rho{self.rho}_diag{self.diag}_mixmean{self.mix_mean}'

        import os
        # create logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        if summary:
            self.summary = SummaryWriter(logdir=f'{self.logdir}')
        self.cnt = 0
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.act = nn.Softplus()

    def save(self, best=False):
        if best:
            torch.save(self.model_list.state_dict(), f'{self.logdir}/best_model_list.pt')

        torch.save(self.model_list.state_dict(), f'{self.logdir}/model_list.pt')

    def load(self, model_path):
        self.model_list.load_state_dict(torch.load(model_path), strict=True)

    def get_L(self):
        L_spatial = self.covariance.L1
        L_temporal = self.covariance.L2

        L_spatial[:, torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = \
            self.act(L_spatial[:, torch.arange(self.num_nodes), torch.arange(self.num_nodes)])
        L_temporal[:, torch.arange(self.num_pred), torch.arange(self.num_pred)] = \
            self.act(L_temporal[:, torch.arange(self.num_pred), torch.arange(self.num_pred)])

        return L_spatial, L_temporal

    def train(self, input, real_val, eval=False):
        if not eval:
            self.model_list.train()

        now = time.time()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        output = output.reshape(output.shape[0], self.num_nodes, self.num_pred, self.n_components)
        print(f"{time.time() - now:.5f}")
        now = time.time()

        mus = output.clone()
        if not self.mix_mean:
            mus = mus[..., 0].unsqueeze(-1)

        R = output
        fc_in = R.permute(0, 3, 2, 1).reshape(-1, self.n_components, self.num_nodes * self.num_pred)
        w = self.fc_w(fc_in)
        w = torch.log_softmax(w, dim=1)

        L_spatial, L_temporal = self.get_L()

        scaled_real_val = self.scaler.transform(real_val)
        print(f"{time.time() - now:.5f}")
        now = time.time()
        if not eval:
            # avoid learning high variance/covariance from missing values
            mask = real_val[:, :, self.pred_len] == 0
            mus = mus * ~(mask.unsqueeze(-1)) + (scaled_real_val[:, :, self.pred_len] * mask).unsqueeze(-1)

        real = real_val[:, :, self.pred_len]

        features = {
            'mu': mus,
            'w': w,
            'L_spatial': L_spatial,
            'L_temporal': L_temporal,
            "rho": self.covariance.rho,
            "target": self.scaler.transform(real),
            "unscaled_target": real,
            "scaler": self.scaler
        }

        loss, nll_loss, mse_loss = self.res_head.forward(
            features=features
        )
        print(f"{time.time() - now:.5f}")
        now = time.time()

        if not eval:
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()
        print(f"{time.time() - now:.5f}")
        now = time.time()

        output = ((mus * w.exp()[..., 0].unsqueeze(1).unsqueeze(1)).sum(-1)).reshape(real_val.shape[0], self.num_nodes, self.num_pred)
        predict = self.scaler.inverse_transform(output)
        predict[predict < 0] = 0
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        mape_list = [util.masked_mape(predict[:, :, i], real[:, :, i], 0.0).item() for i in range(self.num_pred)]
        rmse_list = [util.masked_rmse(predict[:, :, i], real[:, :, i], 0.0).item() for i in range(self.num_pred)]
        mae_list = [util.masked_mae(predict[:, :, i], real[:, :, i], 0.0).item() for i in range(self.num_pred)]

        features["loss"] = loss.item()
        features["nll_loss"] = nll_loss
        features["mse_loss"] = mse_loss
        features["mape"] = mape
        features["rmse"] = rmse
        features["target"] = self.scaler.transform(real)
        features["mape_list"] = mape_list
        features["rmse_list"] = rmse_list
        features["mae_list"] = mae_list

        # if eval:
        #     wandb.log({
        #         "loss_spec/val_loss": loss.item(),
        #         "loss_spec/val_nll_loss": nll_loss,
        #         "loss_spec/val_mape": mape,
        #         "loss_spec/val_rmse": rmse,
        #     })
        # else:
        #     wandb.log({
        #         "loss_spec/train_loss": loss.item(),
        #         "loss_spec/train_nll_loss": nll_loss,
        #         "loss_spec/train_mape": mape,
        #         "loss_spec/train_rmse": rmse,
        #     })

        return features

    def eval(self, input, real_val):
        self.model_list.eval()

        with torch.no_grad():
            info = self.train(input, real_val, eval=True)

        return info
