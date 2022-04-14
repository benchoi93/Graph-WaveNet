import torch.optim as optim
from model import *
import util
import FrEIA


import torch.nn as nn
import torch.distributions as Dist


class LowRankMDNhead(nn.Module):
    def __init__(self, n_components, n_vars, n_rank, pred_len=12):

        self.n_components = n_components
        self.n_vars = n_vars
        self.n_rank = n_rank
        self.pred_len = pred_len

        self.dim_w = n_components
        self.dim_mu = n_components * n_vars
        self.dim_D = n_components * n_vars
        self.dim_V = n_components * n_vars * n_rank

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
        loss = - dist.log_prob(y[:, :, self.pred_len - 1]).mean()
        return loss

    def sample(self, features, n=None):
        dist = self.get_output_distribution(features)

        if n is None:
            return dist.sample()
        else:
            return dist.sample(n)

    def get_output_distribution(self, features):
        # input : features
        # shape of input = (batch_size, hidden)
        w, mu, D, V = self.get_parameters(features)
        mix_dist = Dist.Categorical(w.squeeze(-1))
        com_dist = Dist.LowRankMultivariateNormal(
            loc=torch.einsum('bnc->bcn', mu),
            cov_factor=torch.einsum('bncr->bcnr', V),
            cov_diag=torch.einsum('bnc->bcn', D)
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

        output = torch.einsum("bijk->bjik", output)
        output = output.reshape(-1, self.n_components, self.num_nodes * self.out_per_comp)
        w = self.fc_w(output)
        # w activate softmax0
        w = nn.functional.softmax(w, dim=1)

        loss = self.mdn_head.forward(features={'w': w, 'mu': mus, 'D': D, 'V': V}, y=real_val)
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
