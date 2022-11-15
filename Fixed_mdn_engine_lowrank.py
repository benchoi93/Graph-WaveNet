import torch.optim as optim
from model import *
import util
import math
import properscoring as ps

import seaborn as sns

import torch.nn as nn
import torch.distributions as Dist

from tensorboardX import SummaryWriter


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
        # init_L1 = torch.diag_embed(torch.ones(*self.dim_L_1[:2])) * 0.01
        # init_L2 = torch.diag_embed(torch.ones(*self.dim_L_2[:2])) * 0.01

        self._L1 = nn.Parameter(init_L1.detach(), requires_grad=trainL)
        self._L2 = nn.Parameter(init_L2.detach(), requires_grad=trainL)
        # self.rho = nn.Parameter(torch.ones(1)*rho)
        self.diag = diag
        self.rho = rho

        # mask1 = torch.tril(torch.ones_like(self._L1)).cuda()
        # mask2 = torch.tril(torch.ones_like(self._L2)).cuda()
        # mask2[:, 0, 0] = 0

        # self._L1.register_hook(get_zero_grad_hook(mask1))
        # self._L2.register_hook(get_zero_grad_hook(mask2))

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
        # input : features = dict of tensors
        #    - features['w'] : (batch_size, n_components)
        #    - features['mu'] : (batch_size, n_components, n_vars)
        #    - features['D'] : (batch_size, n_components, n_vars)
        #    - features['V'] : (batch_size, n_components, n_vars, n_rank)
        y = features['target']
        target = y
        # features['target'] = target

        # dist = self.get_output_distribution(features)
        # target = y.reshape(y.shape[0], -1)
        # nll_loss_torch = - dist.log_prob(target).mean()
        # nll_loss = self.get_nll(features, target).mean()
        nll_loss = self.get_nll(features, target).mean()

        reg_loss = 0
        # reg_loss = self.get_sparsity_regularization_loss(features)
        # dist = self.get_output_distribution(features, consider_neighbors=False)

        # min_mse, _ = ((dist.mean - target)**2).min(1)
        # min_mse, _ = ((features["mu"] - target.unsqueeze(1)) ** 2).min(1)
        # mse_loss = ((features["mu"] - target)**2).mean()
        # mse_loss = ((features["mu"] - target).abs()).mean()
        # mae_loss = util.masked_mae(features["mu"], target, 0.0)
        # u_target = features["unscaled_target"]
        # predict = features["scaler"].inverse_transform(features["mu"]).reshape(u_target.shape)
        predict = features["mu"]

        # mse_loss = ((predict - u_target).abs()).mean()
        # mse_loss = util.masked_mse(predict, target, features["scaler"].transform(0.0))
        # mse_loss = self.loss_fn(predict, u_target, 0.0)
        # mse_loss = self.loss_fn(predict, target, features["scaler"].transform(0.0))
        # mse_loss = ((predict - target)**2).mean()
        mse_loss = ((predict - target).abs()).mean()
        # mse_loss = ((features["mu"] - target).abs()).mean()

        loss = self.rho * nll_loss + self.mse_coef * mse_loss
        # loss = mse_loss

        return loss, nll_loss.item(), reg_loss, mse_loss.item(), predict

    def get_nll(self, features, target):
        mu, D, V = self.get_parameters(features)

        # covariance = D + V V^T
        # get NLL of multivariate Gaussian distribution

        b, n, t = mu.shape
        r = V.shape[-1]

        mu = mu.reshape(b, n*t)
        target = target.reshape(b, n*t)
        D = D.reshape(b, n*t)
        V = V.reshape(b, n*t, r)

        Z = target - mu

        # compute inverse of covariance usding woodbury matrix identity
        # inv_cov = (D + V V^T)^-1
        # inv_cov = D^-1 - D^-1 V (I + V^T D^-1 V)^-1 V^T D^-1
        # core = I + V^T D^-1 V
        # logdet_cov = logdet(D) + logdet(core)
        D = torch.nn.functional.softplus(D)
        M = torch.eye(r, device=mu.device) + V.transpose(-1, -2) @ torch.diag_embed(1/D) @ V

        logdet = torch.log(D).sum(-1) + torch.logdet(M)

        # R = Z^T D^-1 V
        R = torch.einsum("bij , bjk -> bik", (Z / D).unsqueeze(1), V)

        # mahabolis1  = Z^T D^-1 Z
        maha1 = (Z/D * Z).sum(1)

        # mahabolis2 = Z^T D^-1 V (I + V^T D^-1 V)^-1 V^T D^-1 Z
        maha2 = (R @ torch.inverse(M) @ R.transpose(-1, -2)).squeeze()

        mahabolis = maha1 - maha2

        nll = mahabolis + logdet
        # nll = n*t * math.log(2 * math.pi) + mahabolis + logdet

        # dist = torch.distributions.LowRankMultivariateNormal(
        #     loc=mu,
        #     cov_factor=V,
        #     cov_diag=D,
        # )
        # nll2 = - dist.log_prob(target)
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
        b, n, t, r = R.shape
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
        return features['mu'], features['D'], features["V"]


class MDN_trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, num_rank, nhid, dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, n_components, reg_coef,
                 mode="cholesky", time_varying=False, consider_neighbors=False, outlier_distribution=True, pred_len=12, rho=0.5, diag=False, mse_coef=0.1, nonlinearity="sigmoid",
                 loss="maskedmse", summary=True):

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

        # dim_out = self.dim_w + self.dim_mu + self.dim_V + self.dim_D
        # self.out_per_comp = 2
        self.mode = mode
        self.time_varying = time_varying
        self.outlier_distribution = outlier_distribution
        self.num_pred = len(self.pred_len) if isinstance(self.pred_len, list) else 1

        self.out_per_comp = num_rank * self.num_pred

        # self.out_per_comp = num_rank + 1
        dim_out = n_components * self.out_per_comp

        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit,
                           in_dim=in_dim, out_dim=dim_out, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        # self.mdn_head = LowRankMDNhead(n_components, num_nodes, num_rank, reg_coef=reg_coef)
        self.res_head = CholeskyResHead(n_components, num_nodes, num_rank, reg_coef=reg_coef, pred_len=pred_len,
                                        consider_neighbors=consider_neighbors, outlier_distribution=outlier_distribution,
                                        mse_coef=mse_coef, rho=rho, loss=loss)
        # dim_w = [batn_components]
        # dims_c = dim_w, dim_mu, dim_U_entries, dim_i
        # self.mdn = FrEIA.modules.GaussianMixtureModel(dims_in, dims_c)
        # self.covariance = FixedMDN(n_components, num_nodes, self.num_pred, rho=rho, diag=diag, trainL=rho != 0)
        self.covariance = FixedResCov(num_rank, num_nodes, self.num_pred, rho=rho, diag=diag, trainL=rho != 0)

        self.model_list = nn.ModuleDict(
            {
                'model': self.model,
                'res_head': self.res_head,
                'covariance': self.covariance,
            }
        )
        self.model_list.to(device)
        # self.model.to(device)
        # self.covariance.to(device)
        # # self.mdn_head.to(device)
        # self.fc_ws.to(device)
        # self.fc_wt.to(device)

        # self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.fc_ws.parameters()) + list(self.fc_wt.parameters()) +
        #                             list(self.covariance.parameters()), lr=lrate, weight_decay=wdecay)
        self.optimizer = optim.Adam(self.model_list.parameters(), lr=lrate, weight_decay=wdecay)
        # self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

        import datetime
        # self.logdir = f'./logs/GWN_MDN_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_N{n_components}_R{num_rank}_reg{reg_coef}_nhid{nhid}_nei{consider_neighbors}'
        self.logdir = f'./logs/GWNMDN_lowrank_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}{loss}loss_N{n_components}_R{num_rank}_reg{reg_coef}_nhid{nhid}_pred{self.num_pred}_rho{rho}_diag{diag}_msecoef{mse_coef}_nlin{nonlinearity}'

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
            # torch.save(self.model.state_dict(), f'{self.logdir}/best_model.pt')
            # torch.save(self.covariance.state_dict(), f'{self.logdir}/best_covariance.pt')
            # torch.save(self.fc_ws.state_dict(), f'{self.logdir}/best_fc_ws.pt')
            # torch.save(self.fc_wt.state_dict(), f'{self.logdir}/best_fc_wt.pt')
            torch.save(self.model_list.state_dict(), f'{self.logdir}/best_model_list.pt')

        # torch.save(self.model.state_dict(), f'{self.logdir}/model.pt')
        # torch.save(self.covariance.state_dict(), f'{self.logdir}/covariance.pt')
        # torch.save(self.fc_ws.state_dict(), f'{self.logdir}/fc_ws.pt')
        # torch.save(self.fc_wt.state_dict(), f'{self.logdir}/fc_wt.pt')
        torch.save(self.model_list.state_dict(), f'{self.logdir}/model_list.pt')

    def load(self, model_path):
        # self.model.load_state_dict(torch.load(model_path), strict=True)
        # self.covariance.load_state_dict(torch.load(cov_path), strict=True)
        # self.fc_ws.load_state_dict(torch.load(fc_ws_path), strict=True)
        # self.fc_wt.load_state_dict(torch.load(fc_wt_path), strict=True)
        self.model_list.load_state_dict(torch.load(model_path), strict=True)

    def get_L(self):
        L_spatial = self.covariance.L1
        L_temporal = self.covariance.L2

        L_spatial[:, torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = \
            self.act(L_spatial[:, torch.arange(self.num_nodes), torch.arange(self.num_nodes)])
        L_temporal[:, torch.arange(self.num_pred), torch.arange(self.num_pred)] = \
            self.act(L_temporal[:, torch.arange(self.num_pred), torch.arange(self.num_pred)])

        # L_temporal[:, 0, 0] = 1
        # L_temporal = L_temporal / L_temporal[:, 0, 0].unsqueeze(-1).unsqueeze(-1)

        return L_spatial, L_temporal

    def train(self, input, real_val, eval=False):
        if not eval:
            self.model_list.train()
            # self.fc_ws.train()
            # self.fc_wt.train()
            # self.covariance.train()

        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)

        output = output.reshape(-1, self.num_nodes, self.num_pred, self.num_rank)

        mus = output[:, :, :, 0]
        D = output[:, :, :, 1]
        V = output[:, :, :, 2:]

        # L_spatial, L_temporal = self.get_L()

        scaled_real_val = self.scaler.transform(real_val)

        if not eval:
            # avoid learning high variance/covariance from missing values
            mask = real_val[:, :, self.pred_len] == 0
            mus = mus * ~(mask) + scaled_real_val[:, :, self.pred_len] * mask

        real = real_val[:, :, self.pred_len]

        features = {
            'mu': mus,
            'D': D,
            'V': V,
            "rho": self.covariance.rho,
            "target": self.scaler.transform(real),
            "unscaled_target": real,
            "scaler": self.scaler
        }

        loss, nll_loss, reg_loss, mse_loss, predict = self.res_head.forward(
            features=features
        )

        if not eval:
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
        # mape = util.masked_mape(predict, real, 0.0).item()
        # rmse = util.masked_rmse(predict, real, 0.0).item()

        # output = self.mdn_head.sample(features={'w': w, 'mu': mus, 'scale_tril': L})
        # output = self.mdn_head.get_output_distribution(features).mean
        # real = real_val[:, :, self.pred_len].reshape(real_val.shape[0], -1)

        # output = predict
        predict = self.scaler.inverse_transform(predict)
        predict[predict < 0] = 0
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        # crps = self.specific_eval(features={'w': w, 'mu': mus, 'scale_tril': L, "rho": self.covariance.rho, 'target': scaled_real_val})

        mape_list = [util.masked_mape(predict[:, :, i], real[:, :, i], 0.0).item() for i in range(self.num_pred)]
        rmse_list = [util.masked_rmse(predict[:, :, i], real[:, :, i], 0.0).item() for i in range(self.num_pred)]
        mae_list = [util.masked_mae(predict[:, :, i], real[:, :, i], 0.0).item() for i in range(self.num_pred)]

        features["loss"] = loss.item()
        features["nll_loss"] = nll_loss
        features["reg_loss"] = reg_loss
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
