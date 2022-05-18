import torch.optim as optim
from model import *
import util
import properscoring as ps

import seaborn as sns

import torch.nn as nn
import torch.distributions as Dist

from tensorboardX import SummaryWriter


class LowRankMDNhead(nn.Module):
    def __init__(self, n_components, n_vars, n_rank, pred_len=12, reg_coef=0.1):

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

    def forward(self, features, y):
        # input : features = dict of tensors
        #    - features['w'] : (batch_size, n_components)
        #    - features['mu'] : (batch_size, n_components, n_vars)
        #    - features['D'] : (batch_size, n_components, n_vars)
        #    - features['V'] : (batch_size, n_components, n_vars, n_rank)

        # check if 'w' , 'mu' , 'D' and 'V' are in features.keys()
        # raise error if not
        assert('w' in features.keys())
        assert('mu' in features.keys())
        assert('D' in features.keys())
        assert('V' in features.keys())

        dist = self.get_output_distribution(features)
        nll_loss = - dist.log_prob(y[:, :, self.pred_len - 1]).mean()
        reg_loss = self.get_sparsity_regularization_loss(dist)
        loss = nll_loss + reg_loss * self.reg_coef
        return loss, nll_loss.item(), reg_loss.item()

    def get_sparsity_regularization_loss(self, dist):
        reg_loss = ((dist.component_distribution.precision_matrix) ** 2).mean()
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
            cov_diag=torch.ones_like(D) * 0.1
        )

        dist = Dist.MixtureSameFamily(mix_dist, com_dist)
        return dist

    def get_parameters(self, features):
        # input : features : dict of tensors, keys: w, mu, D, V
        return features['w'], features['mu'], features['D'], features['V']


class MDN_trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, num_rank, nhid, dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, n_components):

        self.num_nodes = num_nodes
        self.n_components = n_components
        self.num_rank = num_rank

        self.device = device

        self.dim_w = n_components
        self.dim_mu = n_components * num_nodes
        self.dim_V = n_components * num_nodes * num_rank
        self.dim_D = n_components * num_nodes

        # dim_out = self.dim_w + self.dim_mu + self.dim_V + self.dim_D
        self.out_per_comp = (num_rank + 2)
        dim_out = n_components * self.out_per_comp

        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit,
                           in_dim=in_dim, out_dim=dim_out, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.mdn_head = LowRankMDNhead(n_components, num_nodes, num_rank)
        # dim_w = [batn_components]
        # dims_c = dim_w, dim_mu, dim_U_entries, dim_i
        # self.mdn = FrEIA.modules.GaussianMixtureModel(dims_in, dims_c)

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
        # self.mdn_head.to(device)
        self.fc_w.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

        import datetime
        self.logdir = f'./logs/GWN_MDN_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.summary = SummaryWriter(logdir=f'{self.logdir}')
        self.cnt = 0

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)

        output = output.view(-1, self.num_nodes, self.n_components, self.out_per_comp)

        # w = output[:, :, :, :self.dim_w]
        mus = output[:, :, :, 0]
        D = output[:, :, :, 1]
        # D activate ELU
        D = nn.functional.elu(D) + 1

        V = output[:, :, :, 2:]

        mus = torch.einsum('abc->acb', mus)
        D = torch.einsum('abc -> acb', D)
        V = torch.einsum('abcd -> acbd', V)

        output = torch.einsum("bijk->bjik", output)
        output = output.reshape(-1, self.n_components, self.num_nodes * self.out_per_comp)
        w = self.fc_w(output)
        # w activate softmax0
        w = nn.functional.softmax(w, dim=1)

        scaled_real_val = self.scaler.transform(real_val)
        loss, nll_loss, reg_loss = self.mdn_head.forward(features={'w': w, 'mu': mus, 'D': D, 'V': V}, y=scaled_real_val)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # mape = util.masked_mape(predict, real, 0.0).item()
        # rmse = util.masked_rmse(predict, real, 0.0).item()
        real = real_val[:, :, 11]
        output = self.mdn_head.sample(features={'w': w, 'mu': mus, 'D': D, 'V': V})
        predict = self.scaler.inverse_transform(output)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        return loss.item(), mape, rmse, nll_loss, reg_loss

    def eval(self, input, real_val):
        if self.model.training:
            self.specific = True

        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        output = output.view(-1, self.num_nodes, self.n_components, self.out_per_comp)
        mus = output[:, :, :, 0]
        D = output[:, :, :, 1]
        # D activate ELU
        D = nn.functional.elu(D) + 1

        V = output[:, :, :, 2:]

        mus = torch.einsum('abc->acb', mus)
        D = torch.einsum('abc -> acb', D)
        V = torch.einsum('abcd -> acbd', V)

        output = torch.einsum("bijk->bjik", output)
        output = output.reshape(-1, self.n_components, self.num_nodes * self.out_per_comp)
        w = self.fc_w(output)
        # w activate softmax0
        w = nn.functional.softmax(w, dim=1)

        scaled_real_val = self.scaler.transform(real_val)
        loss, nll_loss, reg_loss = self.mdn_head.forward(features={'w': w, 'mu': mus, 'D': D, 'V': V}, y=scaled_real_val)

        # output = self.model(input)
        # output = output.transpose(1, 3)
        #output = [batch_size,12,num_nodes,1]
        # real = torch.unsqueeze(real_val, dim=1)
        real = real_val[:, :, 11]
        output = self.mdn_head.sample(features={'w': w, 'mu': mus, 'D': D, 'V': V})
        predict = self.scaler.inverse_transform(output)
        # predict = output
        # loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        if self.specific:
            self.specific_eval(features={'w': w, 'mu': mus, 'D': D, 'V': V}, y=real_val)

        return loss.item(), mape, rmse, nll_loss, reg_loss

    def specific_eval(self, features, y):
        self.specific = False
        w = features['w']
        mu = features['mu']
        D = features['D']
        V = features['V']

        output = self.mdn_head.sample(features={'w': w, 'mu': mu, 'D': D, 'V': V}, n=1000)
        # pred = self.scaler.inverse_transform(output)
        real_val = y[:, 11, :]
        # real_val = real_val.expand_as(output)

        crps = torch.zeros(size=(y.shape[0], y.shape[2]))
        for i in range(y.shape[0]):
            for j in range(y.shape[2]):
                pred = self.scaler.inverse_transform(output[:, i, j]).cpu().numpy()
                crps[i, j] = ps.crps_ensemble(real_val[i, j].cpu().numpy(), pred)

        self.summary.add_scalar('val/crps', crps.mean().item(), self.cnt)

        dist = self.mdn_head.get_output_distribution(features)
        sample_cov = dist.component_distribution.covariance_matrix[0]

        corr = torch.zeros_like(sample_cov)
        for i in range(sample_cov.size(0)):
            corr[i] = torch.corrcoef(sample_cov[i])

        for i in range(sample_cov.shape[0]):
            sns_plot = sns.heatmap(corr[i].detach().cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
            fig = sns_plot.get_figure()
            self.summary.add_figure('corr_matrix/' + str(i), fig,  self.cnt)

            sns_plot = sns.heatmap(sample_cov[i].detach().cpu().numpy(), cmap='coolwarm')
            fig = sns_plot.get_figure()
            self.summary.add_figure('cov_matrix/' + str(i), fig,  self.cnt)
