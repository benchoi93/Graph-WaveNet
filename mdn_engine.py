import torch.optim as optim
from model import *
import util
import properscoring as ps

import seaborn as sns

import torch.nn as nn
import torch.distributions as Dist

from tensorboardX import SummaryWriter


class MDN_trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, num_rank, nhid, dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, n_components, tau):

        self.num_nodes = num_nodes
        self.n_components = n_components
        self.num_rank = num_rank

        self.device = device

        self.dim_w = n_components
        self.dim_mu = n_components * num_nodes
        self.dim_V = n_components * num_nodes * num_rank
        self.dim_D = n_components * num_nodes

        # dim_out = self.dim_w + self.dim_mu + self.dim_V + self.dim_D
        self.out_per_comp = (num_rank + 1)
        dim_out = n_components * self.out_per_comp

        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit,
                           in_dim=in_dim, out_dim=dim_out, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        # self.mdn_head = LowRankMDNhead(n_components, num_nodes, num_rank, reg_coef=reg_coef)

        self.fc_S = nn.Sequential(
            nn.Linear(num_nodes*n_components, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, num_rank)
        )

        self.model.to(device)
        self.fc_S.to(device)

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.fc_S.parameters()), lr=lrate, weight_decay=wdecay)
        # self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

        import datetime
        self.logdir = f'./logs/GWN_MDN_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_N{n_components}_R{num_rank}_tau{tau}_nhid{nhid}'
        self.summary = SummaryWriter(logdir=f'{self.logdir}')
        self.cnt = 0

    def save(self):
        torch.save(self.model.state_dict(), f'{self.logdir}/model.pt')
        torch.save(self.fc_w.state_dict(), f'{self.logdir}/fc_w.pt')

    def load(self, model_path, fc_w_path):
        self.model.load_state_dict(torch.load(model_path))
        self.fc_w.load_state_dict(torch.load(fc_w_path))

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)

        output = output.view(-1, self.num_nodes, self.n_components, self.out_per_comp)

        mus = output[:, :, :, 0]

        V = output[:, :, :, 1:]
        V = torch.einsum('abcd -> acbd', V)

        V = V.reshape(-1, self.num_nodes * self.n_components, self.num_rank)  # B X NT X R
        # use Woodbury identity to compute the covariance matrix
        # cov = torch.einsum("bij, bjk -> bik", V, V.transpose(-1, -2))  # TODO: plus sigma^2 I

        V = V.view(-1,  self.num_rank,  self.num_nodes * self.n_components)  # B X R X NT
        S = self.fc_S(V)  # B X R X R

        scaled_real_val = self.scaler.transform(real_val)

        # change this part to compute (Y- K_yx K_xx^-1 X)
        ##################
        loss, nll_loss, reg_loss = self.mdn_head.forward(features={'mu': mus, 'V': V}, y=scaled_real_val)
        ##################

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # mape = util.masked_mape(predict, real, 0.0).item()
        # rmse = util.masked_rmse(predict, real, 0.0).item()
        real = real_val
        # output = self.mdn_head.sample(features={'w': w, 'mu': mus, 'D': D, 'V': V})
        predict = self.scaler.inverse_transform(output)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        if self.model.training:
            self.specific = True

        self.model.eval()

        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)

        output = output.view(-1, self.num_nodes, self.n_components, self.out_per_comp)

        mus = output[:, :, :, 0]

        V = output[:, :, :, 1:]
        V = torch.einsum('abcd -> acbd', V)

        scaled_real_val = self.scaler.transform(real_val)

        # TODO: change this part to compute (Y- K_yx K_xx^-1 X)
        ##################
        loss, nll_loss, reg_loss = self.mdn_head.forward(features={'mu': mus, 'V': V}, y=scaled_real_val)
        ##################

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # mape = util.masked_mape(predict, real, 0.0).item()
        # rmse = util.masked_rmse(predict, real, 0.0).item()
        real = real_val
        # output = self.mdn_head.sample(features={'w': w, 'mu': mus, 'D': D, 'V': V})
        predict = self.scaler.inverse_transform(output)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        # TODO: fix this part to draw covariance matrix
        # if self.specific:
        #     self.specific_eval(features={'w': w, 'mu': mus, 'D': D, 'V': V}, y=real_val)

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
