import torch.optim as optim
from model import *
import util
import numpy as np

import torch.nn as nn
import torch.distributions as Dist
from torch.distributions import constraints


class SparseMDNhead(nn.Module):
    def __init__(self, n_components, n_vars, pred_len=12):

        self.n_components = n_components
        self.n_vars = n_vars
        self.pred_len = pred_len

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
        assert('precision' in features.keys())

        # w, mu, precision, U, log_det, info = self.get_output_distribution_params(features)
        # loss = self.calculate_nll_loss(w, mu, precision, U, log_det, info, y)
        # return loss
        dist, info = self.get_output_distribution(features)
        loss = dist.log_prob(y[info, :, self.pred_len-1]).sum(-1).mean()
        return loss

    def calculate_nll_loss(self, w, mu, precision, U, log_det, info, y):
        # input : w, mu, precision, U, log_det, info, y
        # calculate negative log likelihood loss

        x = y[:, :, self.pred_len - 1]
        x = x.unsqueeze(1).expand_as(mu)
        z = (x - mu).unsqueeze(-1)
        z = torch.einsum('bcij,bcjk->bcik', U, z)

        inside = w.squeeze(-1) + log_det - 0.5 * torch.norm(z, dim=-2, p=2).squeeze(-1)
        inside = inside[torch.where(info.sum(1) == 0)[0], :]
        loss = - torch.logsumexp(inside, dim=-1)
        return loss.mean()

    def sample(self, features, n=None):
        dist = self.get_output_distribution(features)

        if n is None:
            return dist.sample()
        else:
            return dist.sample(n)

    def get_output_distribution_params(self, features):
        # input : features
        # shape of input = (batch_size, hidden)
        w, mu, precision = self.get_parameters(features)
        # mix_dist = Dist.Categorical(logits=w.squeeze(-1))
        # com_dist = Dist.LowRankMultivariateNormal(
        #     loc=torch.einsum('bnc->bcn', mu),
        #     cov_factor=torch.einsum('bncr->bcnr', V),
        #     cov_diag=torch.einsum('bnc->bcn', D)
        # )
        U, info = torch.linalg.cholesky_ex(precision, upper=True)
        # log_det = 0.5 * torch.logdet(precision)
        log_det = torch.diagonal(U, dim1=2, dim2=3).log().sum(2)

        # com_dist = Dist.MultivariateNormal(
        #     loc=mu,
        #     precision_matrix=precision
        # )

        # dist = Dist.MixtureSameFamily(mix_dist, com_dist)
        return w, mu, precision, U, log_det, info

    def get_output_distribution(self, features):
        # input : features
        # shape of input = (batch_size, hidden)
        w, mu, precision = self.get_parameters(features)

        # _, info = torch.linalg.cholesky_ex(precision)
        _, info1 = torch.linalg.cholesky_ex(torch.flip(precision, (-2, -1)))
        info1 = info1.sum(1) == 0
        info2 = constraints.positive_definite.check(precision).sum(-1) == w.shape[1]
        info = torch.where(torch.logical_and(info1, info2))[1]

        w = w[info]
        mu = mu[info]
        precision = precision[info]

        mix_dist = Dist.Categorical(logits=w.squeeze(-1))
        com_dist = Dist.MultivariateNormal(
            loc=mu,
            precision_matrix=precision
        )

        dist = Dist.MixtureSameFamily(mix_dist, com_dist)
        return dist, info

    def get_parameters(self, features):
        # input : features : dict of tensors, keys: w, mu, D, V
        return features['w'], features['mu'], features['precision']


class SparseMDN_trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes,
                 num_rank, nhid, dropout, lrate, wdecay, device,
                 supports, gcn_bool, addaptadj, aptinit, n_components, adj_1hop):

        self.num_nodes = num_nodes
        self.n_components = n_components
        self.num_rank = num_rank
        self.adj_1hop = adj_1hop
        self.adj_1hop_idx = adj_1hop.nonzero()

        self.device = device

        # self.dim_w = n_components
        # self.dim_mu = n_components * num_nodes
        # self.dim_V = n_components * num_nodes * num_rank
        # self.dim_D = n_components * num_nodes

        # dim_out = self.dim_w + self.dim_mu + self.dim_V + self.dim_D
        self.out_per_comp = np.sum(adj_1hop) + self.num_nodes + self.num_nodes
        dim_out = n_components * self.out_per_comp

        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit,
                           in_dim=in_dim, out_dim=dim_out, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.mdn_head = SparseMDNhead(n_components, num_nodes)
        # dim_w = [batn_components]
        # dims_c = dim_w, dim_mu, dim_U_entries, dim_i
        # self.mdn = FrEIA.modules.GaussianMixtureModel(dims_in, dims_c)

        self.fc_precision = nn.Sequential(
            nn.Linear(num_nodes*self.out_per_comp, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, self.out_per_comp)
        )
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
        self.fc_precision.to(device)
        self.fc_w.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)

        output = output.reshape(-1, self.n_components, self.out_per_comp * self.num_nodes)
        w = self.fc_w(output)
        output = self.fc_precision(output)

        # diagonal_values = torch.nn.ELU()(output[:, :, :self.num_nodes]) + 1
        diagonal_values = output[:, :, :self.num_nodes].exp() + 1
        mu_values = output[:, :, self.num_nodes:(self.num_nodes * 2)]
        sparse_values = output[:, :, (self.num_nodes*2):]

        precision = torch.zeros(size=(output.shape[0], self.n_components, self.num_nodes, self.num_nodes), device=output.device)
        precision += torch.diag_embed(diagonal_values)
        precision[:, :, self.adj_1hop_idx[0], self.adj_1hop_idx[1]] += sparse_values
        precision[:, :, self.adj_1hop_idx[1], self.adj_1hop_idx[0]] += sparse_values

        y = self.scaler.transform(real_val)

        loss = self.mdn_head.forward(features={'w': w, 'mu': mu_values, 'precision': precision}, y=y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # mape = util.masked_mape(predict, real, 0.0).item()
        # rmse = util.masked_rmse(predict, real, 0.0).item()
        mape = 0
        rmse = 0

        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
