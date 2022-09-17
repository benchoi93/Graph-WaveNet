import torch.optim as optim
from model import *
import util
import properscoring as ps

import seaborn as sns

import torch.nn as nn
import torch.distributions as Dist

from tensorboardX import SummaryWriter


class FixedLowRankMDN(nn.Module):
    def __init__(self, n_components, n_vars, n_rank):
        super(FixedMDN, self).__init__()
        self.dim_D = (n_vars, n_components)
        self.dim_V = (n_vars, n_components, n_rank)

        self.D = nn.Parameter(torch.randn(self.dim_D))
        self.V = nn.Parameter(torch.randn(self.dim_V))


class FixedMDN(nn.Module):
    def __init__(self, n_components, n_vars, rho=0.5, diag=False, trainL=True):
        super(FixedMDN, self).__init__()
        self.dim_L = (n_components, n_vars, n_vars)

        # self.rho = nn.Parameter(torch.ones(1)*rho)
        self.diag = diag
        self.rho = rho

        if diag:
            init_L = torch.ones(*self.dim_L[:2]) * 0.01
            self._L = nn.Parameter(init_L.detach(), requires_grad=trainL)
        else:
            init_L = torch.diag_embed(torch.ones(*self.dim_L[:2])) * 0.01
            self._L = nn.Parameter(init_L.detach(), requires_grad=trainL)

    @property
    def L(self):
        # Ltemp = torch.tanh(self._L) * self.rho
        if self.diag:
            return self._L
        else:
            return torch.tril(self._L)


class LowRankMDNhead(nn.Module):
    def __init__(self, n_components, n_vars, n_rank, pred_len=12, reg_coef=0.1, consider_neighbors=False):
        super(LowRankMDNhead, self).__init__()

        self.n_components = n_components
        self.n_vars = n_vars
        self.n_rank = n_rank
        self.pred_len = pred_len

        self.dim_w = n_components
        self.dim_mu = n_components * n_vars
        self.dim_D = n_components * n_vars
        self.dim_V = n_components * n_vars * n_rank

        self.reg_coef = reg_coef

        self.output_dim = self.dim_w + self.dim_mu + self.dim_D + self.dim_V

        self.consider_neighbors = consider_neighbors

    def forward(self, features, y):
        # input : features = dict of tensors
        #    - features['w'] : (batch_size, n_components)
        #    - features['mu'] : (batch_size, n_components, n_vars)
        #    - features['D'] : (batch_size, n_components, n_vars)
        #    - features['V'] : (batch_size, n_components, n_vars, n_rank)

        dist = self.get_output_distribution(features)

        nll_loss = - dist.log_prob(y[:, :, self.pred_len]).mean()

        reg_loss = 0
        # reg_loss = self.get_sparsity_regularization_loss(dist)

        target = y[:, :, self.pred_len - 1]
        # min_mse, _ = ((dist.mean - target)**2).min(1)
        # min_mse, _ = ((features["mu"] - target.unsqueeze(1)) ** 2).min(1)
        mse_loss = ((dist.mean - target)**2).mean(1).mean()

        # loss = nll_loss + reg_loss * self.reg_coef + mse_loss*100
        loss = features["rho"] * nll_loss + reg_loss * self.reg_coef
        # loss = mse_loss

        return loss, nll_loss.item(), reg_loss, mse_loss.item()

    def get_sparsity_regularization_loss(self, dist):
        # reg_loss = ((dist.component_distribution.precision_matrix) ** 2).mean()
        precision = dist.precision_matrix
        b, d, _ = precision.size()
        # get LASSO for non-diagonal elements of precision matrices
        reg_loss = precision.flatten(1)[:, 1:].view(b, d-1, d+1)[:, :, :-1].abs().mean()
        return reg_loss

    def sample(self, features, n=None):
        dist = self.get_output_distribution(features)

        if n is None:
            return dist.sample()
        else:
            return dist.sample((n,))

    def get_output_distribution(self, features):
        # input : features
        # shape of input = (batch_size, hidden)
        w, mu, D, V = self.get_parameters(features)
        mix_dist = Dist.Categorical(w.squeeze(-1))
        com_dist = Dist.LowRankMultivariateNormal(
            loc=mu,
            cov_factor=V,
            # cov_factor=torch.zeros_like(V),
            # cov_diag=torch.ones_like(D) * 0.1
            # cov_diag=torch.ones_like(D) * 0.01
            cov_diag=D
        )

        dist = Dist.MixtureSameFamily(mix_dist, com_dist)
        return dist

    def get_parameters(self, features):
        # check if 'w' , 'mu' , 'D' and 'V' are in features.keys()
        # raise error if not
        assert('w' in features.keys())
        assert('mu' in features.keys())
        assert('D' in features.keys())
        assert('V' in features.keys())

        # input : features : dict of tensors, keys: w, mu, D, V
        return features['w'], features['mu'], features['D'], features['V']


class CholeskyMDNhead(LowRankMDNhead):
    def __init__(self, n_components, n_vars, n_rank, pred_len=12, reg_coef=0.1,
                 consider_neighbors=False, outlier_distribution=False,
                 outlier_distribution_kwargs=None, mse_coef=0.1, rho=0.1, diag=False):

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
        self.diag = diag

    def forward(self, features):
        # input : features = dict of tensors
        #    - features['w'] : (batch_size, n_components)
        #    - features['mu'] : (batch_size, n_components, n_vars)
        #    - features['D'] : (batch_size, n_components, n_vars)
        #    - features['V'] : (batch_size, n_components, n_vars, n_rank)
        y = features['target']
        target = y[:, :, self.pred_len]
        # features['target'] = target

        dist = self.get_output_distribution(features)
        target = y[:, :, self.pred_len].reshape(y.shape[0], -1)
        nll_loss = - dist.log_prob(target).mean() if self.rho != 0 else 0

        reg_loss = 0 if self.reg_coef != 0 else 0  # TODO
        # reg_loss = self.get_sparsity_regularization_loss(dist)
        # dist = self.get_output_distribution(features)

        # min_mse, _ = ((dist.mean - target)**2).min(1)
        # min_mse, _ = ((features["mu"] - target.unsqueeze(1)) ** 2).min(1)
        mse_loss = ((features["mu"] - target)**2).mean()

        loss = self.rho * nll_loss + reg_loss * self.reg_coef + self.mse_coef * mse_loss
        # loss = mse_loss

        if isinstance(nll_loss, torch.Tensor):
            nll_loss = nll_loss.item()
        if isinstance(reg_loss, torch.Tensor):
            reg_loss = reg_loss.item()
        if isinstance(mse_loss, torch.Tensor):
            mse_loss = mse_loss.item()

        return loss, nll_loss, reg_loss, mse_loss

    def get_output_distribution(self, features):
        if self.diag:
            return self.get_output_distribution_diag(features)
        else:
            return self.get_output_distribution_cholesky(features)

    def get_output_distribution_diag(self, features):
        w, mu, scale = self.get_parameters(features)

        # sum of w_i * scale_tril_i
        b, c, d = scale.size()
        scale_new = torch.zeros((b, d), device=scale.device)

        for i in range(c):
            scale_new += w[:, i].unsqueeze(-1) * scale[:, i, :]

        dist = Dist.Normal(
            loc=mu,
            scale=scale_new
        )

        diag_dist = Dist.Independent(dist, 1)

        return diag_dist

    def get_output_distribution_cholesky(self, features, consider_neighbors=False):
        # input : features
        # shape of input = (batch_size, hidden)
        # w, mu, cov = self.get_parameters(features)
        w, mu, scale_tril = self.get_parameters(features)

        # sum of w_i * scale_tril_i
        b, c, d, _ = scale_tril.size()
        scale_tril_new = torch.zeros((b, d, d), device=scale_tril.device)

        for i in range(c):
            scale_tril_new += w[:, i].unsqueeze(-1).unsqueeze(-1) * scale_tril[:, i, :, :]

        dist = Dist.MultivariateNormal(
            loc=mu,
            scale_tril=scale_tril_new
        )

        return dist

    def get_parameters(self, features):
        # input : features : dict of tensors, keys: w, mu, D, V
        # check if 'w' , 'mu' , 'L' are in features.keys()
        # raise error if not
        assert('w' in features.keys())
        assert('mu' in features.keys())
        # assert('cov' in features.keys())
        # return features['w'], features['mu'], features['cov']
        return features['w'], features['mu'], features['scale_tril']


class MDN_trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, num_rank, nhid, dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, n_components, reg_coef,
                 mode="cholesky", time_varying=False, consider_neighbors=False, outlier_distribution=True, pred_len=12, rho=0.5, diag=False, mse_coef=0.1):

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

        self.out_per_comp = num_rank + self.num_pred

        # self.out_per_comp = num_rank + 1
        dim_out = n_components * self.out_per_comp

        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit,
                           in_dim=in_dim, out_dim=dim_out, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        # self.mdn_head = LowRankMDNhead(n_components, num_nodes, num_rank, reg_coef=reg_coef)
        self.mdn_head = CholeskyMDNhead(n_components, num_nodes, num_rank, reg_coef=reg_coef, pred_len=pred_len,
                                        consider_neighbors=consider_neighbors, outlier_distribution=outlier_distribution,
                                        mse_coef=mse_coef, diag=diag, rho=rho)
        # dim_w = [batn_components]
        # dims_c = dim_w, dim_mu, dim_U_entries, dim_i
        # self.mdn = FrEIA.modules.GaussianMixtureModel(dims_in, dims_c)
        self.covariance = FixedMDN(n_components, num_nodes * self.num_pred, rho=rho, diag=diag, trainL=rho != 0)

        self.fc_w = nn.Sequential(
            nn.Linear(self.n_components*num_nodes*self.out_per_comp, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, self.n_components)
        )

        self.model.to(device)
        self.covariance.to(device)
        # self.mdn_head.to(device)
        self.fc_w.to(device)

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.fc_w.parameters()) +
                                    list(self.covariance.parameters()), lr=lrate, weight_decay=wdecay)
        # self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

        import datetime
        # self.logdir = f'./logs/GWN_MDN_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_N{n_components}_R{num_rank}_reg{reg_coef}_nhid{nhid}_nei{consider_neighbors}'
        self.logdir = f'./logs/GWNMDN_multistep_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_N{n_components}_R{num_nodes}_reg{reg_coef}_nhid{nhid}_pred{pred_len}_rho{rho}_diag{diag}_msecoef{mse_coef}'

        self.summary = SummaryWriter(logdir=f'{self.logdir}')
        self.cnt = 0
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        # self.softplus = nn.Softplus(dim=-1)

    def save(self, best=False):
        if best:
            torch.save(self.model.state_dict(), f'{self.logdir}/best_model.pt')
            torch.save(self.covariance.state_dict(), f'{self.logdir}/best_covariance.pt')
            torch.save(self.fc_w.state_dict(), f'{self.logdir}/best_fc_w.pt')

        torch.save(self.model.state_dict(), f'{self.logdir}/model.pt')
        torch.save(self.covariance.state_dict(), f'{self.logdir}/covariance.pt')
        torch.save(self.fc_w.state_dict(), f'{self.logdir}/fc_w.pt')

    def load(self, model_path, cov_path, fc_w_path):
        self.model.load_state_dict(torch.load(model_path))
        self.covariance.load_state_dict(torch.load(cov_path))
        self.fc_w.load_state_dict(torch.load(fc_w_path))

    def train(self, input, real_val, eval=False):
        if not eval:
            self.mdn_head.training = True
            self.model.train()
            self.fc_w.train()
            self.covariance.train()

        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)

        # output = output.view(-1, self.num_nodes, self.n_components, self.out_per_comp)

        if self.diag:
            L = self.covariance.L.unsqueeze(0).expand(output.shape[0], -1, -1)
        else:
            L = torch.tril(self.covariance.L.unsqueeze(0).expand(output.shape[0], -1, -1, -1))
        mus = output[:, 0, :, :self.num_pred]
        mus = mus.reshape(-1, self.num_pred * self.num_nodes)

        if self.covariance.rho != 0:
            if self.diag:
                L = torch.nn.functional.elu(L) + 1
            else:
                L[:, :, torch.arange(L.shape[-1]), torch.arange(L.shape[-1])] =\
                    torch.nn.functional.elu(L[:, :, torch.arange(L.shape[-1]), torch.arange(L.shape[-1])]) + 1

        output = output.reshape(-1, self.n_components*self.num_nodes * self.out_per_comp)
        w = self.fc_w(output)
        w = self.softmax(w)

        scaled_real_val = self.scaler.transform(real_val)
        loss, nll_loss, reg_loss, mse_loss = self.mdn_head.forward(
            features={'w': w, 'mu': mus, 'scale_tril': L, "rho": self.covariance.rho, 'target': scaled_real_val})

        if not eval:
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
        # mape = util.masked_mape(predict, real, 0.0).item()
        # rmse = util.masked_rmse(predict, real, 0.0).item()
        real = real_val[:, :, self.pred_len].reshape(real_val.shape[0], -1)

        # output = self.mdn_head.sample(features={'w': w, 'mu': mus, 'scale_tril': L})
        output = self.mdn_head.get_output_distribution(
            features={'w': w, 'mu': mus, 'scale_tril': L, "rho": self.covariance.rho, 'target': scaled_real_val}).mean
        predict = self.scaler.inverse_transform(output)
        predict[predict < 0] = 0
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        # crps = self.specific_eval(features={'w': w, 'mu': mus, 'scale_tril': L, "rho": self.covariance.rho, 'target': scaled_real_val})

        info = {
            "w": w,
            "mu": mus,
            "scale_tril": L,
            "loss": loss.item(),
            "mape": mape,
            "rmse": rmse,
            "nll_loss": nll_loss,
            "reg_loss": reg_loss,
            "mse_loss": mse_loss,
            # "crps": crps
            "target": self.scaler.transform(real)
        }

        return info

    def eval(self, input, real_val):
        self.model.eval()
        self.fc_w.eval()
        self.covariance.eval()

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
        mask_ES = mask.sum(-1) != 0
        mask_crps = mask.reshape(crps.shape)

        ES = (ES * (1 - mask_ES.float())).mean()
        crps = (crps * (1-mask_crps.float()).cpu().numpy()).mean()

        return crps, ES

    def get_crps(self, features):
        # output = self.mdn_head.sample(features=features, n=100)
        # output = [self.mdn_head.sample(features=features, n=1) for i in range(100)]
        output = self.mdn_head.sample(features=features, n=100)
        # output = torch.concat(output, dim=0)
        scaled_real_val = features['target']
        real_val = self.scaler.inverse_transform(scaled_real_val)
        # real_val = real_val[:, :, self.pred_len]

        # pred = self.scaler.inverse_transform(output)
        # real_val = y[:, :, 11]
        # real_val = real_val.expand_as(output)
        s, b, _ = output.shape
        n = self.num_nodes
        t = self.num_pred
        beta = 1
        output = output.reshape(b, s, n*t)
        output = output.reshape(b, s, n, t)
        output = self.scaler.inverse_transform(output)
        output[output < 0] = 0
        real_val = real_val.reshape(b, n, t)

        # mask = real_val == 0

        diff = (output-real_val.unsqueeze(1)).abs()
        diff = torch.norm(diff.reshape(b, s, n*t), p=2, dim=-1)
        cdist = torch.cdist(output.reshape(b, s, n*t), output.reshape(b, s, n*t), p=2)

        ES = diff.mean(-1) - (cdist.sum((-1, -2))/(s**2) * 0.5)
        # ES[mask.sum((-1, -2)) == 0].mean()
        # crps = torch.zeros(size=(b, n, t))
        # for i in range(b):
        #     for j in range(n):
        #         for k in range(t):
        #             pred = output[:, i, j, k].cpu().numpy()
        #             pred[pred < 0] = 0
        #             # crps_empirical = ps.crps_ensemble(real_val[i, j, k].cpu().numpy(), pred)
        #             # sigma_space = L_spatial[:, j, j]
        #             # sigma_temp = L_temporal[:, k, k]
        #             # sigma = (omega[i, :] * sigma_space * sigma_temp).sum() * self.scaler.std
        #             crps_normal = ps.crps_gaussian(real_val[i, j, k].cpu().numpy(), mu=pred.mean(), sig=pred.std())
        #             crps[i, j, k] = crps_normal
        # mu = output.mean(1)
        # sigma = output.std(1)
        # crps = ps.crps_gaussian(real_val.cpu().numpy(), mu=mu.cpu().numpy(), sig=sigma.cpu().numpy())
        crps = ps.crps_ensemble(real_val.cpu().numpy(), output.cpu().numpy(), axis=1)
        # crps.mean((-1, -2))[mask.cpu().numpy().sum((-1, -2)) == 0].mean()
        return crps, ES
        # self.cnt += 1

    def plot_cov(self, features):
        # dist = self.mdn_head.get_output_distribution(features)
        # sample_cov = dist.component_distribution.covariance_matrix[0]
        # sample_prec = dist.component_distribution.precision_matrix[0]
        sample_cov = torch.einsum("bij,bjk->bik", self.covariance.L, self.covariance.L.transpose(-1, -2))

        corr = torch.zeros_like(sample_cov)
        for i in range(sample_cov.size(0)):
            corr[i] = torch.corrcoef(sample_cov[i])

        # sparsity =V (sample_prec.abs() > 0.01).float()

        for i in range(sample_cov.shape[0]):
            sns_plot = sns.heatmap(corr[i].detach().cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
            fig = sns_plot.get_figure()
            self.summary.add_figure('corr_matrix/' + str(i), fig,  self.cnt)

            sns_plot = sns.heatmap(sample_cov[i].detach().cpu().numpy(), cmap='coolwarm')
            fig = sns_plot.get_figure()
            self.summary.add_figure('cov_matrix/' + str(i), fig,  self.cnt)

            # sns_plot = sns.heatmap(sample_prec[i].detach().cpu().numpy(), cmap='coolwarm')
            # fig = sns_plot.get_figure()
            # self.summary.add_figure('prec_matrix/' + str(i), fig,  self.cnt)

            # sns_plot = sns.heatmap(sparsity[i].detach().cpu().numpy(), cmap='coolwarm')
            # fig = sns_plot.get_figure()
            # self.summary.add_figure('sparsity/' + str(i), fig,  self.cnt)

        self.cnt += 1
