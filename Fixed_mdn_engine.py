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
    def __init__(self, n_components, n_vars):
        super(FixedMDN, self).__init__()
        self.dim_L = (n_components, n_vars, n_vars)

        self.L = nn.Parameter(torch.tril(torch.randn(*self.dim_L)))


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

        nll_loss = - dist.log_prob(y[:, :, self.pred_len - 1]).mean()
        if self.consider_neighbors:
            nll_loss = - dist.log_prob(y[:, :, self.pred_len - 2]).mean()
            nll_loss += - dist.log_prob(y[:, :, self.pred_len]).mean()

        reg_loss = self.get_sparsity_regularization_loss(dist)
        loss = nll_loss + reg_loss * self.reg_coef
        return loss, nll_loss.item(), reg_loss.item()

    def get_sparsity_regularization_loss(self, dist):
        # reg_loss = ((dist.component_distribution.precision_matrix) ** 2).mean()
        precision = dist.component_distribution.precision_matrix
        b, n, d, _ = precision.size()
        # get LASSO for non-diagonal elements of precision matrices
        reg_loss = precision.flatten(2)[:, :, 1:].view(b, n, d-1, d+1)[:, :, :, :-1].abs().mean()
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
    def __init__(self, n_components, n_vars, n_rank, pred_len=12, reg_coef=0.1, consider_neighbors=False, outlier_distribution=True, outlier_distribution_kwargs=None):
        super(CholeskyMDNhead, self).__init__(n_components, n_vars, n_rank, pred_len, reg_coef, consider_neighbors)

        self.n_components = n_components
        self.n_vars = n_vars
        self.n_rank = n_rank
        self.pred_len = pred_len

        self.reg_coef = reg_coef
        self.consider_neighbors = consider_neighbors
        self.outlier_distribution = outlier_distribution

        if outlier_distribution:
            self.outlier_w = 1e-3
            self.outlier_distribution_mu = 0
            self.outlier_distribution_sigma = 2

        self.training = True

    def get_output_distribution(self, features):
        # input : features
        # shape of input = (batch_size, hidden)
        # w, mu, cov = self.get_parameters(features)
        w, mu, scale_tril = self.get_parameters(features)

        if self.outlier_distribution and self.training:
            b = w.size(0)
            w = torch.cat([w, self.outlier_w*torch.ones((b, 1, 1), device=w.device)], dim=1)
            w = w / w.sum(1, keepdim=True)

            mu = torch.cat([mu, self.outlier_distribution_mu * torch.ones((b, 1, self.n_vars), device=mu.device)], dim=1)

            scale_tril = torch.cat([scale_tril, torch.diag_embed(self.outlier_distribution_sigma *
                                                                 torch.ones((b, 1, self.n_vars), device=scale_tril.device))], dim=1)

        mix_dist = Dist.Categorical(w.squeeze(-1))
        com_dist = Dist.MultivariateNormal(
            loc=mu,
            scale_tril=scale_tril
        )

        dist = Dist.MixtureSameFamily(mix_dist, com_dist)
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
                 mode="cholesky", time_varying=False, consider_neighbors=False, outlier_distribution=True):

        self.num_nodes = num_nodes
        self.n_components = n_components
        self.num_rank = num_rank

        self.device = device

        self.dim_w = n_components
        self.dim_mu = n_components * num_nodes
        self.dim_V = n_components * num_nodes * num_rank
        self.dim_D = n_components * num_nodes

        # dim_out = self.dim_w + self.dim_mu + self.dim_V + self.dim_D
        # self.out_per_comp = 2
        self.mode = mode
        self.time_varying = time_varying
        self.outlier_distribution = outlier_distribution

        if time_varying:
            if mode == "cholesky":
                self.out_per_comp = num_rank + 1
            elif mode == "lowrank":
                self.out_per_comp = 1
        else:
            if mode == "cholesky":
                self.out_per_comp = num_rank + 1
            elif mode == "lowrank":
                raise NotImplementedError

        # self.out_per_comp = num_rank + 1
        dim_out = n_components * self.out_per_comp

        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit,
                           in_dim=in_dim, out_dim=dim_out, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        # self.mdn_head = LowRankMDNhead(n_components, num_nodes, num_rank, reg_coef=reg_coef)
        self.mdn_head = CholeskyMDNhead(n_components, num_nodes, num_rank, reg_coef=reg_coef,
                                        consider_neighbors=consider_neighbors, outlier_distribution=outlier_distribution)
        # dim_w = [batn_components]
        # dims_c = dim_w, dim_mu, dim_U_entries, dim_i
        # self.mdn = FrEIA.modules.GaussianMixtureModel(dims_in, dims_c)
        self.covariance = FixedMDN(n_components, num_nodes)

        self.fc_w = nn.Sequential(
            nn.Linear(num_nodes*self.out_per_comp, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, 1)
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
        self.logdir = f'./logs/GWN_MDN_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_N{n_components}_R{num_rank}_reg{reg_coef}_nhid{nhid}_nei{consider_neighbors}'
        self.summary = SummaryWriter(logdir=f'{self.logdir}')
        self.cnt = 0

    def save(self):
        torch.save(self.model.state_dict(), f'{self.logdir}/model.pt')
        torch.save(self.covariance.state_dict(), f'{self.logdir}/covariance.pt')
        torch.save(self.fc_w.state_dict(), f'{self.logdir}/fc_w.pt')

    def load(self, model_path, cov_path, fc_w_path):
        self.model.load_state_dict(torch.load(model_path))
        self.covariance.load_state_dict(torch.load(cov_path))
        self.fc_w.load_state_dict(torch.load(fc_w_path))

    def train(self, input, real_val):
        self.mdn_head.training = True
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)

        output = output.view(-1, self.num_nodes, self.n_components, self.out_per_comp)
        L = torch.tril(self.covariance.L.unsqueeze(0).expand(output.shape[0], -1, -1, -1))
        L[:, :, torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = torch.nn.functional.elu(
            L[:, :, torch.arange(self.num_nodes), torch.arange(self.num_nodes)]) + 1

        mus = output[:, :, :, 0]

        V = output[:, :, :, 1:]
        V = torch.einsum('abcd -> acbd', V)

        # if self.time_varying:
        #     covariance = torch.zeros((output.shape[0], self.n_components, self.num_nodes, self.num_nodes), device=self.device)
        #     for i in range(self.num_rank):
        #         temp = torch.einsum('bcij, bcjk->bcik', L, V[:, :, :, i:(i+1)])
        #         covariance += torch.einsum('bcij, bcik->bcik', temp, temp.transpose(-1, 2))
        #     covariance /= self.num_rank
        # else:
        #     covariance = torch.einsum("bcij , bcjk -> bcik", L, L.transpose(-1, -2))

        # D = self.covariance.D.unsqueeze(0).expand(output.shape[0], -1, -1)
        # V = self.covariance.V.unsqueeze(0).expand(output.shape[0], -1, -1, -1)
        # output = torch.cat([output, D.unsqueeze(-1), V], -1)

        mus = torch.einsum('abc->acb', mus)
        # D = torch.einsum('abc -> acb', D)
        # V = torch.einsum('abcd -> acbd', V)
        # D = nn.functional.elu(D) + 1
        output = output.reshape(-1, self.n_components, self.num_nodes * self.out_per_comp)
        w = self.fc_w(output)
        # w activate softmax0
        w = nn.functional.softmax(w, dim=1)

        scaled_real_val = self.scaler.transform(real_val)
        loss, nll_loss, reg_loss = self.mdn_head.forward(features={'w': w, 'mu': mus, 'scale_tril': L}, y=scaled_real_val)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # mape = util.masked_mape(predict, real, 0.0).item()
        # rmse = util.masked_rmse(predict, real, 0.0).item()
        real = real_val[:, :, 11]
        # output = self.mdn_head.sample(features={'w': w, 'mu': mus, 'scale_tril': L})
        output = self.mdn_head.get_output_distribution(features={'w': w, 'mu': mus, 'scale_tril': L}).mean
        predict = self.scaler.inverse_transform(output)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        return loss.item(), mape, rmse, nll_loss, reg_loss

    def eval(self, input, real_val):
        self.mdn_head.training = False
        if self.model.training:
            self.specific = True

        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)

        output = output.view(-1, self.num_nodes, self.n_components, self.out_per_comp)
        L = torch.tril(self.covariance.L.unsqueeze(0).expand(output.shape[0], -1, -1, -1))
        L[:, :, torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = torch.nn.functional.elu(
            L[:, :, torch.arange(self.num_nodes), torch.arange(self.num_nodes)]) + 1

        mus = output[:, :, :, 0]

        V = output[:, :, :, 1:]
        V = torch.einsum('abcd -> acbd', V)

        # if self.time_varying:
        #     covariance = torch.zeros((output.shape[0], self.n_components, self.num_nodes, self.num_nodes), device=self.device)
        #     for i in range(self.num_rank):
        #         temp = torch.einsum('bcij, bcjk->bcik', L, V[:, :, :, i:(i+1)])
        #         covariance += torch.einsum('bcij, bcik->bcik', temp, temp.transpose(-1, 2))
        #     covariance /= self.num_rank
        # else:
        #     covariance = torch.einsum("bcij , bcjk -> bcik", L, L.transpose(-1, -2))

        # D = self.covariance.D.unsqueeze(0).expand(output.shape[0], -1, -1)
        # V = self.covariance.V.unsqueeze(0).expand(output.shape[0], -1, -1, -1)
        # output = torch.cat([output, D.unsqueeze(-1), V], -1)

        mus = torch.einsum('abc->acb', mus)
        # D = torch.einsum('abc -> acb', D)
        # V = torch.einsum('abcd -> acbd', V)
        # D = nn.functional.elu(D) + 1
        output = output.reshape(-1, self.n_components, self.num_nodes * self.out_per_comp)
        w = self.fc_w(output)
        # w activate softmax0
        w = nn.functional.softmax(w, dim=1)

        scaled_real_val = self.scaler.transform(real_val)
        loss, nll_loss, reg_loss = self.mdn_head.forward(features={'w': w, 'mu': mus, 'scale_tril': L}, y=scaled_real_val)

        # output = self.model(input)
        # output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        # real = torch.unsqueeze(real_val, dim=1)
        real = real_val[:, :, 11]
        output = self.mdn_head.get_output_distribution(features={'w': w, 'mu': mus, 'scale_tril': L}).mean
        predict = self.scaler.inverse_transform(output)
        # predict = output
        # loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        if self.specific:
            self.specific_eval(features={'w': w, 'mu': mus, 'scale_tril': L}, y=real_val)

        return loss.item(), mape, rmse, nll_loss, reg_loss

    def specific_eval(self, features, y):
        self.specific = False
        w = features['w']
        mu = features['mu']
        L = features['scale_tril']
        # V = features['V']

        output = self.mdn_head.sample(features={'w': w, 'mu': mu, 'scale_tril': L}, n=1000)
        # pred = self.scaler.inverse_transform(output)
        real_val = y[:, :, 11]
        # real_val = real_val.expand_as(output)

        crps = torch.zeros(size=(y.shape[0], y.shape[2]))
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                pred = self.scaler.inverse_transform(output[:, i, j]).cpu().numpy()
                crps[i, j] = ps.crps_ensemble(real_val[i, j].cpu().numpy(), pred)

        self.summary.add_scalar('val/crps', crps.mean().item(), self.cnt)

        dist = self.mdn_head.get_output_distribution(features)
        sample_cov = dist.component_distribution.covariance_matrix[0]
        sample_prec = dist.component_distribution.precision_matrix[0]

        corr = torch.zeros_like(sample_cov)
        for i in range(sample_cov.size(0)):
            corr[i] = torch.corrcoef(sample_cov[i])

        sparsity = (sample_prec.abs() > 0.01).float()

        for i in range(sample_cov.shape[0]):
            sns_plot = sns.heatmap(corr[i].detach().cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
            fig = sns_plot.get_figure()
            self.summary.add_figure('corr_matrix/' + str(i), fig,  self.cnt)

            sns_plot = sns.heatmap(sample_cov[i].detach().cpu().numpy(), cmap='coolwarm')
            fig = sns_plot.get_figure()
            self.summary.add_figure('cov_matrix/' + str(i), fig,  self.cnt)

            sns_plot = sns.heatmap(sample_prec[i].detach().cpu().numpy(), cmap='coolwarm')
            fig = sns_plot.get_figure()
            self.summary.add_figure('prec_matrix/' + str(i), fig,  self.cnt)

            sns_plot = sns.heatmap(sparsity[i].detach().cpu().numpy(), cmap='coolwarm')
            fig = sns_plot.get_figure()
            self.summary.add_figure('sparsity/' + str(i), fig,  self.cnt)

        self.cnt += 1
