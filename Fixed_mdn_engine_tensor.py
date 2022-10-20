import pickle
import numpy as np
import torch.optim as optim
from model import *
import util
import math
import properscoring as ps
import scipy
import seaborn as sns

import torch.nn as nn
import torch.distributions as Dist
import tensorly as tl
from tensorly.decomposition import parafac
from tensorboardX import SummaryWriter

tl.set_backend("pytorch")
# init_L1 = torch.rand((n_vars, num_rank))
# init_L2 = torch.rand((num_pred, num_rank))
# init_L3 = torch.rand((num_obs, num_rank))

# out = tl.cp_tensor.cp_to_tensor((None, [init_L3, init_L1, init_L2]))
# out.shape
# DATA_X = np.load("/app/data/PEMS-BAY/train.npz")['x'][:, :, :, 0]
# factors = parafac(DATA_X, rank=50, n_iter_max=100, init='random', verbose=True)
# recon = tl.cp_tensor.cp_to_tensor(factors)
# np.sqrt(np.abs((DATA_X - recon)).mean())


def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask
    return hook


class FixedResCov(nn.Module):
    def __init__(self, n_components, n_vars, num_pred, num_rank, diag=False, trainL=True, device='cpu', factor=None):
        super(FixedResCov, self).__init__()
        self.dim_L_1 = (n_components, n_vars, num_rank)
        self.dim_L_2 = (n_components, num_pred, num_rank)

        if factor is None:
            init_L1 = torch.rand(*self.dim_L_1)
            init_L2 = torch.rand(*self.dim_L_2)

            self._L1 = nn.Parameter(init_L1.detach(), requires_grad=trainL)
            self._L2 = nn.Parameter(init_L2.detach(), requires_grad=trainL)

        else:
            self._L1 = nn.Parameter(torch.from_numpy(factor[1][2]).float().detach().unsqueeze(0), requires_grad=False)
            self._L2 = nn.Parameter(torch.from_numpy(factor[1][1]).float().detach().unsqueeze(0), requires_grad=False)

    @property
    def L1(self):
        return self._L1

    @property
    def L2(self):
        return self._L2


class CholeskyResHead(nn.Module):
    def __init__(self, n_components, n_vars, n_rank, pred_len=12, reg_coef=0.1,
                 consider_neighbors=False, outlier_distribution=False,
                 outlier_distribution_kwargs=None, mse_coef=0.1, rho=0.1, loss="maskedmse"):
        super(CholeskyResHead, self).__init__()
        self.n_components = n_components
        self.n_vars = n_vars
        self.n_rank = n_rank
        self.pred_len = pred_len

        self.mse_coef = mse_coef
        self.rho = rho

        self.reg_coef = reg_coef
        self.consider_neighbors = consider_neighbors
        self.outlier_distribution = outlier_distribution

        if outlier_distribution:
            self.outlier_w = 1e-3
            self.outlier_distribution_mu = 0
            self.outlier_distribution_sigma = 2

        self.training = True

        if loss == "maskedmse":
            self.loss_fn = util.masked_mse
            # self.loss_fn = torch.nn.MSELoss(reduction='mean')
        elif loss == "maskedmae":
            self.loss_fn = util.masked_mae
            # self.loss_fn = torch.nn.L1Loss(reduction='mean')

    def forward(self, features):
        y = features['target']
        target = y
        # target = y.reshape(y.shape[0], -1)
        # nll_loss = self.get_nll(features, target).mean()

        mu = features['mu']
        R = features['R']
        L_spatial = features['L1']
        L_temporal = features['L2']
        rho = features['rho']

        # b, c, r = R.shape
        # n = mu.shape[1]
        # t = mu.shape[2]

        # R_unscaled = features["scaler"].inverse_transform(R)
        # mu_unscaled = features["scaler"].inverse_transform(mu)

        # cp_residual = torch.stack([tl.cp_tensor.cp_to_tensor((None, [R[:, i, :], L_temporal[i], L_spatial[i]])) for i in range(c)], 1)
        cp_residual = torch.einsum("ar,br,cr->abc", R, L_temporal[0],  L_spatial[0])
        cp_residual = cp_residual.permute(0, 2, 1)

        # pred_unscaled = mu_unscaled + cp_residual.sum(1)
        pred_unscaled = cp_residual
        target_unscaled = features['unscaled_target']

        pred = features["scaler"].transform(pred_unscaled) + mu

        mask = features['mask']
        normp = 2

        unscaled_target = features['unscaled_target']

        input_a = tl.tenalg.khatri_rao([L_spatial[0], L_temporal[0]])
        target_a = tl.unfold(unscaled_target, mode=0)

        R_actual = torch.linalg.solve(input_a.T @ input_a, input_a.T @ target_a.T).T
        # reconstruction = tl.cp_tensor.cp_to_tensor((None, (R_actual, L_spatial[0], L_temporal[0])))

        # mse_loss_res = torch.norm((target - mu - cp_residual.sum(1)) * (~mask), p=normp) / torch.norm((~mask).float(), p=normp)
        # mse_loss_mu = torch.norm((target - cp_residual.sum(1)) * (~mask), p=normp) / torch.norm((~mask).float(), p=normp)

        # mse_loss = rho * mse_loss_res + (1-rho) * mse_loss_mu
        mse_loss = util.masked_mae(pred_unscaled, target_unscaled, 0.0)
        # mse_loss = torch.norm((target - mu) * (~mask), p=1)

        reg_loss = (R_actual - R).abs().mean()
        # mureg_loss = ((mu**2).mean((0, 1)) * torch.exp(np.log(1.1)*(torch.arange(1, 13))).to(mu.device)).sum()
        # mureg_loss = ((mu**2).mean((0, 1))).sum()
        mureg_loss = ((mu**2).mean((0, 1))).sum()

        # loss = self.rho * nll_loss + reg_loss * self.reg_coef + self.mse_coef * mse_loss

        loss = reg_loss + self.rho * mse_loss + self.reg_coef * mureg_loss
        # loss = mse_loss

        return loss, reg_loss.item(), mse_loss.item(), mureg_loss.item(), pred

    def get_nll(self, features, target):
        mu, R, L_s, L_t = self.get_parameters(features)

        b, nt, r = R.shape
        n = self.n_vars
        t = len(self.pred_len)

        R_ext = torch.concat([R, (target - mu - R.sum(2)).unsqueeze(-1)], dim=2)
        R_flatten = R_ext.reshape(b, r+1, nt).reshape(b, r+1, n, t)

        U_inv = torch.einsum("rij, rjk ->rik", L_s, L_s.transpose(-1, -2))
        V_inv = torch.einsum("rij, rjk ->rik", L_t, L_t.transpose(-1, -2))
        # U_inv = U_inv.unsqueeze(0).repeat(b, 1, 1, 1)
        # V_inv = V_inv.unsqueeze(0).repeat(b, 1, 1, 1)

        L_t = L_t.unsqueeze(0).repeat(b, 1, 1, 1)
        L_s = L_s.unsqueeze(0).repeat(b, 1, 1, 1)

        # mahabolis = -0.5 * torch.einsum("brij,brjk,brkl,brlp->brip", V_inv, R_flatten.transpose(-1, -2), U_inv, R_flatten)
        # mahabolis = mahabolis.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        Ulogdet = torch.logdet(U_inv)
        Vlogdet = torch.logdet(V_inv)
        Ulogdet = L_s.diagonal(dim1=-1, dim2=-2).log().sum(-1) * 2
        Vlogdet = L_t.diagonal(dim1=-1, dim2=-2).log().sum(-1) * 2

        Q_t = torch.einsum("brij,brjk,brkl->bril", L_s.transpose(-1, -2), R_flatten, L_t)
        mahabolis = -0.5 * torch.pow(Q_t, 2).sum((-1, -2))

        nll = -(-n*t/2 * math.log(2*math.pi) + mahabolis + n/2 * Vlogdet + t/2 * Ulogdet)
        return nll

    def get_output_distribution(self, features, consider_neighbors=False):
        # input : features
        # shape of input = (batch_size, hidden)
        # w, mu, cov = self.get_parameters(features)
        ws, wt, mu, L_spatial, L_temporal = self.get_parameters(features)
        prc_spatial_new, prc_temporal_new = self.get_mnd(features)

        prc = util.kron(prc_spatial_new, prc_temporal_new)
        # cov = cov.unsqueeze(0).expand(mu.shape[0], -1, -1, -1)
        # cov = (w.unsqueeze(-1).unsqueeze(-1).expand_as(cov) * cov).sum(1)

        dist = Dist.MultivariateNormal(
            loc=mu,
            # scale_tril=scale_tril_new
            # covariance_matrix=cov_new
            precision_matrix=prc
        )

        return dist

    def sample(self, features, nsample=None):
        # dist = self.get_output_distribution(features)
        mu, R, L_s, L_t = self.get_parameters(features)
        b, nt, r = R.shape
        n = self.n_vars
        t = len(self.pred_len)

        # U_inv = torch.einsum("rij, rjk ->rik", L_s, L_s.transpose(-1, -2))
        # V_inv = torch.einsum("rij, rjk ->rik", L_t, L_t.transpose(-1, -2))
        # U_inv = U_inv.unsqueeze(0).repeat(b, 1, 1, 1)
        # V_inv = V_inv.unsqueeze(0).repeat(b, 1, 1, 1)

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
        # input : features : dict of tensors, keys: w, mu, D, V
        # check if 'w' , 'mu' , 'L' are in features.keys()
        # raise error if not
        # assert('w' in features.keys())
        # assert('mu' in features.keys())
        # assert('cov' in features.keys())
        # return features['w'], features['mu'], features['cov']
        return features['mu'], features['R'], features["L_spatial"], features["L_temporal"]


class MDN_trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, num_rank, nhid, dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, n_components, reg_coef,
                 mode="cholesky", time_varying=False, consider_neighbors=False, outlier_distribution=True, pred_len=12, rho=0.5, diag=False, mse_coef=0.1, nonlinearity="sigmoid",
                 loss="maskedmse", summary=True, factor=None):

        self.num_nodes = num_nodes
        self.n_components = n_components
        self.num_rank = num_rank

        self.device = device
        self.diag = diag

        self.dim_w = n_components
        self.dim_mu = n_components * num_nodes
        self.dim_V = n_components * num_nodes * num_rank
        self.dim_D = n_components * num_nodes
        self.pred_len = pred_len
        self.rho = rho

        # dim_out = self.dim_w + self.dim_mu + self.dim_V + self.dim_D
        # self.out_per_comp = 2
        self.mode = mode
        self.time_varying = time_varying
        self.outlier_distribution = outlier_distribution
        self.num_pred = len(self.pred_len) if isinstance(self.pred_len, list) else 1

        self.out_per_comp = (num_rank+1) * self.num_pred

        # self.out_per_comp = num_rank + 1
        # dim_out = n_components * self.out_per_comp
        dim_out = nhid*8

        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit,
                           in_dim=in_dim, out_dim=dim_out, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.mu_model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit,
                              in_dim=in_dim, out_dim=self.num_pred, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        # self.mdn_head = LowRankMDNhead(n_components, num_nodes, num_rank, reg_coef=reg_coef)
        self.res_head = CholeskyResHead(n_components, num_nodes, num_rank, reg_coef=reg_coef, pred_len=pred_len,
                                        consider_neighbors=consider_neighbors, outlier_distribution=outlier_distribution,
                                        mse_coef=mse_coef, rho=rho, loss=loss)
        # dim_w = [batn_components]
        # dims_c = dim_w, dim_mu, dim_U_entries, dim_i
        # self.mdn = FrEIA.modules.GaussianMixtureModel(dims_in, dims_c)
        # self.covariance = FixedMDN(n_components, num_nodes, self.num_pred, rho=rho, diag=diag, trainL=rho != 0)
        self.covariance = FixedResCov(n_components, num_nodes, self.num_pred, self.num_rank, diag=diag, trainL=rho != 0, factor=factor)

        self.fc_out = nn.Sequential(
            # nn.Linear(num_nodes*self.num_pred*num_rank, nhid),
            nn.Linear(dim_out, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, num_rank)
        )

        self.model_list = nn.ModuleDict(
            {
                'model': self.model,
                'res_head': self.res_head,
                'covariance': self.covariance,
                'fc_out': self.fc_out
            }
        )
        self.model_list.to(device)
        self.mu_model.to(device)

        # self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.fc_ws.parameters()) + list(self.fc_wt.parameters()) +
        #                             list(self.covariance.parameters()), lr=lrate, weight_decay=wdecay)
        self.optimizer = optim.Adam(self.model_list.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizer_mu = optim.Adam(self.mu_model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

        import datetime
        # self.logdir = f'./logs/GWN_MDN_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_N{n_components}_R{num_rank}_reg{reg_coef}_nhid{nhid}_nei{consider_neighbors}'
        self.logdir = f'./logs/GWNMDN_tensor_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}{loss}loss_N{n_components}_R{num_rank}_reg{reg_coef}_nhid{nhid}_pred{self.num_pred}_rho{rho}_diag{diag}_msecoef{mse_coef}_lr{lrate}'

        if summary:
            self.summary = SummaryWriter(logdir=f'{self.logdir}')
        self.cnt = 0
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-1)

        if nonlinearity == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif nonlinearity == "softplus":
            self.act = nn.Softplus()
        elif nonlinearity == "elu":
            def elusplusone(x): return nn.ELU()(x) + 1
            self.act = elusplusone
        elif nonlinearity == "sigmoid":
            self.act = nn.Sigmoid()
        elif nonlinearity == "exp":
            def exp(x): return torch.exp(x)
            self.act = exp
        # self.softplus = nn.Softplus()
        # self.softmax = nn.Softmax(dim=-1)

    def save(self, best=False):
        if best:
            torch.save(self.model_list.state_dict(), f'{self.logdir}/best_model_list.pt')

        torch.save(self.model_list.state_dict(), f'{self.logdir}/model_list.pt')

    def load(self, model_path):
        self.model_list.load_state_dict(torch.load(model_path), strict=True)

    def train(self, input, real_val, eval=False):
        if not eval:
            self.model_list.train()
            self.mu_model.train()
            # self.fc_ws.train()
            # self.fc_wt.train()
            # self.covariance.train()

        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        # output = output.reshape(-1, self.n_components, self.num_nodes, self.num_pred, (self.num_rank+1))
        # output = output.reshape(-1, self.n_components, self.num_nodes * self.num_pred, (self.num_rank+1))

        mus = self.mu_model(input)
        mus = mus.transpose(1, 3).squeeze(1)

        fc_in = output.squeeze(1)
        # mus = mus.reshape(-1, self.n_components, self.num_nodes, self.num_pred)
        # fc_in = output[:, :, :, 1:]

        fc_in = fc_in.mean(1)
        R = self.fc_out(fc_in)

        scaled_real_val = self.scaler.transform(real_val)
        # if not eval:
        #     # avoid learning high variance/covariance from missing values
        #     mask = real_val[:, :, self.pred_len] == 0
        #     mus = mus * ~(mask).reshape(mus.shape) + scaled_real_val[:, :, self.pred_len].reshape(mus.shape) * mask.reshape(mus.shape)
        real = real_val[:, :, self.pred_len]
        mask = real == 0

        features = {
            'mu': mus,
            'R': R,
            'L1': self.covariance.L1,
            'L2': self.covariance.L2,
            "rho": self.rho,
            "target": self.scaler.transform(real),
            "unscaled_target": real,
            "scaler": self.scaler,
            "mask": mask
        }

        loss, res_loss, mse_loss, mu_reg, pred = self.res_head.forward(
            features=features
        )

        if not eval:
            self.optimizer.zero_grad()
            self.optimizer_mu.zero_grad()
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.optimizer_mu.step()
        # mape = util.masked_mape(predict, real, 0.0).item()
        # rmse = util.masked_rmse(predict, real, 0.0).item()

        # output = self.mdn_head.sample(features={'w': w, 'mu': mus, 'scale_tril': L})
        # output = self.mdn_head.get_output_distribution(features).mean
        # real = real_val[:, :, self.pred_len].reshape(real_val.shape[0], -1)

        with torch.no_grad():
            # output = mus.reshape(real_val.shape[0], self.num_nodes, self.num_pred)
            # predict = self.scaler.inverse_transform(output)
            # predict[predict < 0] = 0
            # mape = util.masked_mape(predict, real, 0.0).item()
            # rmse = util.masked_rmse(predict, real, 0.0).item()
            # mae = util.masked_mae(predict, real, 0.0).item()

            output2 = pred.reshape(real_val.shape[0], self.num_nodes, self.num_pred)
            predict2 = self.scaler.inverse_transform(output2)
            predict2[predict2 < 0] = 0
            mape2 = util.masked_mape(predict2, real, 0.0).item()
            rmse2 = util.masked_rmse(predict2, real, 0.0).item()
            mae2 = util.masked_mae(predict2, real, 0.0).item()

            mape_specific = (predict2 - real).abs() / real
            mape_specific = (mape_specific * ~mask).nansum(dim=(0, 1)) / (~mask).sum(dim=(0, 1))
            mape_specific = mape_specific.cpu().numpy()

            rmse_specific = (predict2 - real).pow(2)
            rmse_specific = (rmse_specific * ~mask).nansum(dim=(0, 1)) / (~mask).sum(dim=(0, 1))
            rmse_specific = rmse_specific.sqrt().cpu().numpy()

            mae_specific = (predict2 - real).abs()
            mae_specific = (mae_specific * ~mask).nansum(dim=(0, 1)) / (~mask).sum(dim=(0, 1))
            mae_specific = mae_specific.cpu().numpy()

        # crps = self.specific_eval(features={'w': w, 'mu': mus, 'scale_tril': L, "rho": self.covariance.rho, 'target': scaled_real_val})

        features["loss"] = loss.item()
        features["res_loss"] = res_loss
        features["mse_loss"] = mse_loss
        features["mu_reg"] = mu_reg
        features["mape"] = mape2 * 100
        features["rmse"] = rmse2
        features["mae"] = mae2
        features["mape2"] = mape_specific * 100
        features["rmse2"] = rmse_specific
        features["mae2"] = mae_specific
        features["target"] = self.scaler.transform(real)
        features["pred"] = predict2
        # # "cov": cov_new,
        # "loss": loss.item(),
        # "mape": mape,
        # "rmse": rmse,
        # "nll_loss": nll_loss,
        # "reg_loss": reg_loss,
        # "mse_loss": mse_loss,
        # # "crps": crps
        # "target": self.scaler.transform(real)

        return features

    def eval(self, input, real_val):
        self.model_list.eval()

        with torch.no_grad():
            info = self.train(input, real_val, eval=True)

        # crps, ES = self.specific_eval(features=info)
        # info['crps'] = crps
        # info["ES"] = ES.cpu().numpy()

        info['crps'] = 0
        info["ES"] = 0

        return info

    def specific_eval(self, features):
        crps, ES = self.get_crps(features)

        scaled_real_val = features['target']
        real_val = self.scaler.inverse_transform(scaled_real_val)
        mask = real_val == 0
        # ES = ES[mask.sum(-1) == 0].mean()
        # crps = crps.reshape(crps.shape[0], crps.shape[1] * crps.shape[2])
        # crps = (crps * (~mask.cpu().numpy())).sum() / (~mask.cpu().numpy()).sum()
        # crps.mean((-1, -2))[mask.cpu().numpy().sum((-1)) == 0].mean()

        mask_ES = mask.sum(-1) != 0
        mask_crps = mask.reshape(crps.shape)
        ES = (ES * (1 - mask_ES.float())).mean()
        crps = (crps * (1-mask_crps.float()).cpu().numpy()).mean()

        return crps, ES

    def get_crps(self, features):
        # output = self.mdn_head.sample(features=features, n=100)
        # output = [self.mdn_head.sample(features=features, n=1) for i in range(100)]
        output = self.res_head.sample(features=features, nsample=100)
        # output = torch.concat(output, dim=0)
        scaled_real_val = features['target']
        real_val = self.scaler.inverse_transform(scaled_real_val)
        # real_val = real_val[:, :, self.pred_len]

        # pred = self.scaler.inverse_transform(output)
        # real_val = y[:, :, 11]
        # real_val = real_val.expand_as(output)
        s, b, n, t = output.shape
        beta = 1
        # output = output.reshape(s, b, n, t)
        output = output.reshape(b, s, n, t)
        output = self.scaler.inverse_transform(output)
        output[output < 0] = 0
        real_val = real_val.reshape(b, n, t)

        diff = (output-real_val.unsqueeze(1)).abs()
        diff = torch.norm(diff.reshape(b, s, n*t), p=2, dim=-1)
        cdist = torch.cdist(output.reshape(b, s, n*t), output.reshape(b, s, n*t), p=2)

        ES = diff.mean(-1) - (cdist.sum((-1, -2))/(s**2) * 0.5)

        # import time
        # t1= time.time()
        # crps = torch.zeros(size=(b, n, t))
        # for i in range(b):
        #     for j in range(n):
        #         for k in range(t):
        #             pred = output[:, i, j, k].cpu().numpy()
        #             pred[pred < 0] = 0
        #             crps_empirical = ps.crps_ensemble(real_val[i, j, k].cpu().numpy(), pred)
        #             # sigma_space = L_spatial[:, j, j]
        #             # sigma_temp = L_temporal[:, k, k]
        #             # sigma = (omega[i, :] * sigma_space * sigma_temp).sum() * self.scaler.std
        #             # crps_normal = ps.crps_gaussian(real_val[i, j, k].cpu().numpy(), mu=pred.mean(), sig=pred.std())
        #             crps[i, j, k] = crps_empirical
        # t2 = time.time()
        # print("crps time: ", t2-t1)

        # mu = output.mean(1)
        # sigma = output.std(1)
        # crps = ps.crps_gaussian(real_val.cpu().numpy(), mu=mu.cpu().numpy(), sig=sigma.cpu().numpy())

        crps = ps.crps_ensemble(real_val.cpu().numpy(), output.cpu().numpy(), axis=1)

        return crps, ES
        # self.cnt += 1

    def plot_cov(self):
        # dist = self.mdn_head.get_output_distribution(features)
        # sample_cov = dist.component_distribution.covariance_matrix[0]
        # sample_prec = dist.component_distribution.precision_matrix[0]
        L_spatial, L_temporal = self.get_L()

        sample_cov_spatial = torch.einsum("bij,bjk->bik", L_spatial, L_spatial.transpose(-1, -2))
        sample_cov_temporal = torch.einsum("bij,bjk->bik", L_temporal, L_temporal.transpose(-1, -2))

        sample_cov_spatial = torch.inverse(sample_cov_spatial)
        sample_cov_temporal = torch.inverse(sample_cov_temporal)

        corr_spatial = torch.zeros_like(sample_cov_spatial)
        for i in range(sample_cov_spatial.size(0)):
            corr_spatial[i] = torch.corrcoef(sample_cov_spatial[i])

        corr_temporal = torch.zeros_like(sample_cov_temporal)
        for i in range(sample_cov_spatial.size(0)):
            corr_temporal[i] = torch.corrcoef(sample_cov_temporal[i])

        # sparsity =V (sample_prec.abs() > 0.01).float()

        for i in range(sample_cov_spatial.shape[0]):
            sns_plot = sns.heatmap(corr_spatial[i].detach().cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
            fig = sns_plot.get_figure()
            self.summary.add_figure('spatial_corr_matrix/' + str(i), fig,  self.cnt)

            sns_plot = sns.heatmap(sample_cov_spatial[i].detach().cpu().numpy(), cmap='coolwarm')
            fig = sns_plot.get_figure()
            self.summary.add_figure('spatial_cov_matrix/' + str(i), fig,  self.cnt)

            sns_plot = sns.heatmap(corr_temporal[i].detach().cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
            fig = sns_plot.get_figure()
            self.summary.add_figure('temporal_corr_matrix/' + str(i), fig,  self.cnt)

            sns_plot = sns.heatmap(sample_cov_temporal[i].detach().cpu().numpy(), cmap='coolwarm')
            fig = sns_plot.get_figure()
            self.summary.add_figure('temporal_cov_matrix/' + str(i), fig,  self.cnt)

        self.cnt += 1
