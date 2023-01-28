import torch.optim as optim
from model import *
import util
import math

import torch.nn as nn
import torch.distributions as Dist

from tensorboardX import SummaryWriter


def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask
    return hook


class FixedResCov(nn.Module):
    def __init__(self, n_components, n_vars, num_pred, rho=0.5, diag=False, trainL=True, device='cpu', adj_bool=None):
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
    def __init__(self, n_components, n_vars, n_rank, pred_len=12, rho=0.1, loss="mse"):
        super(CholeskyResHead, self).__init__()
        self.n_components = n_components
        self.n_vars = n_vars
        self.n_rank = n_rank
        self.pred_len = pred_len

        self.rho = rho
        self.training = True

        self.loss = loss

    def forward(self, features):
        target = features['target']
        nll_loss = self.get_nll(features, target).mean()

        predict = features["mu"]

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

        loss = self.rho * nll_loss + mse_loss
        return loss, nll_loss.item(), mse_loss.item()

    def get_nll(self, features, target):
        mu, R, L_s, L_t = self.get_parameters(features)
        w = features["w"]
        logw = w.log().squeeze(-1)

        b, n, t, r = R.shape
        n = self.n_vars
        t = len(self.pred_len)

        R_ext = target - mu
        R_flatten = R_ext.unsqueeze(1).repeat(1, r+1, 1, 1)

        L_t = L_t.unsqueeze(0).repeat(b, 1, 1, 1)  # L_t L_tT = prc_T
        L_s = L_s.unsqueeze(0).repeat(b, 1, 1, 1)  # L_s L_sT = prc_S

        Ulogdet = L_s.diagonal(dim1=-1, dim2=-2).log().sum(-1)
        Vlogdet = L_t.diagonal(dim1=-1, dim2=-2).log().sum(-1)

        Q_t = torch.einsum("brij,brjk,brkl->bril", L_s.transpose(-1, -2), R_flatten, L_t)
        mahabolis = -0.5 * torch.pow(Q_t, 2).sum((-1, -2))

        nll = -n*t/2 * math.log(2*math.pi) + mahabolis + n * Vlogdet + t * Ulogdet + logw

        nll = - torch.logsumexp(nll, dim=1)

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
        return features['mu'], features['R'], features["L_spatial"], features["L_temporal"]


class MDN_trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, num_rank, nhid, dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, n_components,                  mode="cholesky", time_varying=False, pred_len=list(range(12)), rho=0.5, diag=False,
                 loss="maskedmse", summary=True, adj_bool=None):

        self.num_nodes = num_nodes
        self.n_components = n_components
        self.num_rank = num_rank

        self.device = device
        self.diag = diag

        self.dim_w = n_components
        self.dim_mu = n_components * num_nodes
        self.pred_len = pred_len

        adj_bool = adj_bool

        # self.out_per_comp = 2
        self.mode = mode
        self.time_varying = time_varying
        self.num_pred = len(self.pred_len) if isinstance(self.pred_len, list) else 1

        self.out_per_comp = num_rank * self.num_pred

        # self.out_per_comp = num_rank + 1
        dim_out = n_components * self.out_per_comp

        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit,
                           in_dim=in_dim, out_dim=dim_out, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)

        self.res_head = CholeskyResHead(n_components, num_nodes, num_rank, pred_len=pred_len, rho=rho, loss=loss)

        self.covariance = FixedResCov(num_rank, num_nodes, self.num_pred, rho=rho, diag=diag, trainL=rho != 0, adj_bool=adj_bool)

        self.fc_w = nn.Sequential(
            nn.Linear(self.num_nodes * self.num_pred, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, 1)
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

        self.optimizer = optim.Adam(self.model_list.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaler = scaler
        self.clip = 5

        import datetime
        self.logdir = f'./logs/GWN_DynMix_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}{loss}loss_N{n_components}_R{num_rank}_nhid{nhid}_pred{self.num_pred}_rho{rho}_diag{diag}'

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

        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)

        output = output.reshape(output.shape[0], self.num_nodes, self.num_pred, self.num_rank)
        fc_in = output.permute(0, 3, 2, 1).reshape(-1, self.num_rank, self.num_nodes * self.num_pred)
        w = self.fc_w(fc_in)
        w = torch.softmax(w, dim=1)

        mus = output[:, :, :, 0]
        R = output[:, :, :, 1:]

        L_spatial, L_temporal = self.get_L()

        scaled_real_val = self.scaler.transform(real_val)

        if not eval:
            # avoid learning high variance/covariance from missing values
            mask = real_val[:, :, self.pred_len] == 0
            mus = mus * ~(mask) + scaled_real_val[:, :, self.pred_len] * mask

        real = real_val[:, :, self.pred_len]

        features = {
            'mu': mus,
            'w': w,
            'R': R,
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

        if not eval:
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

        output = mus.reshape(real_val.shape[0], self.num_nodes, self.num_pred)
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

        return features

    def eval(self, input, real_val):
        self.model_list.eval()

        with torch.no_grad():
            info = self.train(input, real_val, eval=True)

        return info
