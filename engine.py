import math
import torch.optim as optim
from model import *
import util
import torch.nn as nn


class covariance(nn.Module):
    def __init__(self, num_nodes, delay, device, n_components=1, rho=1, train_L_space=True, train_L_batch=True):
        super(covariance, self).__init__()

        self.n_components = n_components
        self.num_nodes = num_nodes
        self.delay = delay
        self._L_space = nn.Parameter(torch.diag_embed(torch.ones(n_components, num_nodes)).detach(), requires_grad=train_L_space).to(device)
        self._L_batch = nn.Parameter(torch.diag_embed(torch.ones(n_components, delay)).detach(), requires_grad=train_L_batch).to(device)

        self.act = nn.Softplus()

    @property
    def L_space(self):
        return torch.tril(self._L_space)

    @property
    def L_batch(self):
        return torch.tril(self._L_batch)


class batch_opt(nn.Module):
    def __init__(self, num_nodes, delay, rho=1):
        super(batch_opt, self).__init__()

        self.num_nodes = num_nodes
        self.delay = delay
        self.rho = rho

    def get_nll(self, pred, target, L_list, logw=None):
        R = (pred - target)
        Bd, K, N, Q = R.shape
        R = R.reshape(Bd//self.delay, self.delay, K, N, Q)
        R = R.permute(0, 2, 1, 3, 4).squeeze(-1)  # TODO:: just for now. only for single-step prediction

        Ls, Lb = L_list
        L_s = Ls.unsqueeze(0).repeat(Bd//self.delay, 1, 1, 1)
        L_b = Lb.unsqueeze(0).repeat(Bd//self.delay, 1, 1, 1)

        Prc_s_logdet = L_s.diagonal(dim1=-1, dim2=-2).log().sum(-1)
        Prc_b_logdet = L_b.diagonal(dim1=-1, dim2=-2).log().sum(-1)

        Q = torch.einsum("brij,brjk,brkl->bril", L_b.transpose(-1, -2), R, L_s)
        mahabolis = -0.5 * torch.pow(Q, 2).sum((-1, -2))

        n = self.num_nodes
        t = self.delay

        nll = - n*t * math.log(2*math.pi) + mahabolis + n * Prc_b_logdet + t * Prc_s_logdet
        # nll = - n*t * math.log(2*math.pi) + mahabolis + n * Prc_b_logdet + t * Prc_s_logdet + logw
        nll = -torch.logsumexp(nll, dim=-1)

        return nll

    def masked_mse(self, pred, target, mask):
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        mse_loss = (pred-target) ** 2
        mse_loss = mse_loss[:, 0, :, :]  # TODO: just for now for 1 component

        mse_loss = mse_loss * mask
        mse_loss = torch.where(torch.isnan(mse_loss), torch.zeros_like(mse_loss), mse_loss)
        mse_loss = torch.mean(mse_loss)
        return mse_loss

    def loss(self, pred, target, L_list, mask):

        if self.rho == 0:
            return self.masked_mse(pred, target, mask)
        elif self.rho == 1:
            return self.get_nll(pred, target, L_list)
        else:
            nll = self.get_nll(pred, target, L_list)
            mse = self.masked_mse(pred, target, mask)
            return self.rho * nll + (1-self.rho) * mse


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate,
                 wdecay, device, supports, gcn_bool, addaptadj, aptinit, delay, **kwargs):

        self.model = gwnet(device, num_nodes, dropout,
                           supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                           aptinit=aptinit, in_dim=in_dim, out_dim=1,
                           residual_channels=nhid, dilation_channels=nhid,
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.covariance = covariance(num_nodes, delay, device, **kwargs)

        # self.loss = util.masked_mae
        self.loss_class = batch_opt(num_nodes, delay)
        self.loss = self.loss_class.loss

        self.model_list = nn.ModuleDict(
            {
                'model': self.model,
                'covariance': self.covariance
            }
        )

        self.optimizer = optim.Adam(self.model_list.parameters(), lr=lrate, weight_decay=wdecay)

        self.scaler = scaler
        self.clip = 5

        self.act = nn.Softplus()

    def update_diagonal(self, L):
        N, D, _ = L.shape
        L[:, torch.arange(D), torch.arange(D)] = self.act(L[:, torch.arange(D), torch.arange(D)])
        return L

    def get_L(self):
        Ls = self.update_diagonal(self.covariance.L_space)
        Lb = self.update_diagonal(self.covariance.L_batch)

        return Ls, Lb

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        mask = real_val != 0.0

        L_list = self.get_L()

        loss = self.loss(predict, real, L_list, mask).mean()
        # loss = self.loss(predict, real, 0.0).mean()
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        mask = real_val != 0.0
        L_list = self.get_L()
        loss = self.loss(predict, real, L_list, mask).mean()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
