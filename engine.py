import math
import torch.optim as optim
from model import *
import util
import torch.nn as nn


def unfold(tens, mode, dims, align=2):
    """
    Unfolds tensor into matrix.

    Parameters
    ----------
    tens : ndarray, tensor with shape == dims
    mode : int, which axis to move to the front
    dims : list, holds tensor shape

    Returns
    -------
    matrix : ndarray, shape (dims[mode], prod(dims[/mode]))
    """
    # if mode == 0:
    #     return tens.reshape(dims[0], -1)
    # else:
    #     return np.moveaxis(tens, mode, 0).reshape(dims[mode], -1)
    return torch.moveaxis(tens, mode + align, align).reshape(list(tens.shape[:align]) + [dims[mode], -1])


def refold(vec, mode, dims, align=2):
    """
    Refolds vector into tensor.

    Parameters
    ----------
    vec : ndarray, tensor with len == prod(dims)
    mode : int, which axis was unfolded along.
    dims : list, holds tensor shape

    Returns
    -------
    tens : ndarray, tensor with shape == dims
    """

    tens = vec.reshape(list(vec.shape[:align]) + [dims[mode]] + [d for m, d in enumerate(dims) if m != mode])
    return torch.moveaxis(tens, align, mode + align)


def kron_vec_prod(As, vt, align=2):
    """
    Computes matrix-vector multiplication between
    matrix kron(As[0], As[1], ..., As[N]) and vector
    v without forming the full kronecker product.
    """
    dims = [A.shape[-1] for A in As]
    # vt = v.reshape([v.shape[0], v.shape[1]] + dims)
    for i, A in enumerate(As):
        # temp = A @ unfold(vt, i, dims)
        temp = torch.einsum('bnij,bnjk->bnik', A, unfold(vt, i, dims))
        vt = refold(temp, i, dims)
    return vt


class covariance(nn.Module):
    def __init__(self, num_nodes, delay, pred_len, device, n_components=1, train_L_space=True, train_L_time=True, train_L_batch=True):
        super(covariance, self).__init__()

        self.n_components = n_components
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.delay = delay

        self.device = device

        self._L_space = nn.Parameter(torch.zeros(n_components, num_nodes, num_nodes).detach(), requires_grad=train_L_space)
        self._L_time = nn.Parameter(torch.zeros(n_components, pred_len, pred_len).detach(), requires_grad=train_L_time)
        self._L_batch = nn.Parameter(torch.zeros(n_components, delay, delay).detach(), requires_grad=train_L_batch)

        self.elu = torch.nn.ELU()
        self.act = lambda x: self.elu(x) + 1

    @property
    def L_space(self):
        return torch.tril(self._L_space)

    @property
    def L_batch(self):
        return torch.tril(self._L_batch)

    @property
    def L_time(self):
        return torch.tril(self._L_time)

    def update_diagonal(self, L):
        N, D, _ = L.shape
        L[:, torch.arange(D), torch.arange(D)] = self.act(L[:, torch.arange(D), torch.arange(D)])
        return L

    def get_L(self):
        Ls = self.update_diagonal(self.L_space).to(self.device)
        Lt = self.update_diagonal(self.L_time).to(self.device)
        Lb = self.update_diagonal(self.L_batch).to(self.device)
        return Lb, Ls, Lt


class batch_opt(nn.Module):
    def __init__(self, num_nodes, delay, rho=1, det="mse"):
        super(batch_opt, self).__init__()

        self.num_nodes = num_nodes
        self.delay = delay
        self.rho = rho
        self.det = det

    def get_nll_2d(self, pred, target, L_list, logw=None):
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

    def get_nll(self, mu, target, L_list):
        b, d, n, t = mu.shape

        mu = mu.unsqueeze(1)

        # mask = (target != 0)
        R_ext = (mu - target)
        # R_flatten = R_ext.reshape(b, d * n * t).unsqueeze(1)
        # R_ext = R_ext.unsqueeze(1)

        L_list = [l.transpose(-1, -2).unsqueeze(0).repeat(b, 1, 1, 1) for l in L_list]
        logdet = [l.diagonal(dim1=-1, dim2=-2).log().sum(-1) for l in L_list]

        L_x = kron_vec_prod(L_list, R_ext, align=2)
        mahabolis = -0.5 * L_x.pow(2).sum((-1, -2, -3))

        dnt = d * n * t
        logdet = sum([dnt*ll/L_list[i].shape[-1] for i, ll in enumerate(logdet)])

        nll = -dnt/2 * math.log(2*math.pi) + mahabolis + logdet

        nll = - torch.logsumexp(nll, dim=1)
        return nll

    def masked_mse(self, pred, target, mask):
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        pred = pred.unsqueeze(1)

        if self.det == "mse":
            mse_loss = (pred-target) ** 2
        elif self.det == "mae":
            mse_loss = torch.abs(pred-target)
        else:
            raise NotImplementedError

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
                 wdecay, device, supports, gcn_bool, addaptadj, aptinit, delay,
                 train_L_space, train_L_time, train_L_batch, rho, det):

        self.model = gwnet(device, num_nodes, dropout,
                           supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                           aptinit=aptinit, in_dim=in_dim, out_dim=seq_length,
                           residual_channels=nhid, dilation_channels=nhid,
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.covariance = covariance(num_nodes, delay, seq_length, device,
                                     train_L_space=train_L_space,
                                     train_L_time=train_L_time,
                                     train_L_batch=train_L_batch)

        # self.loss = util.masked_mae
        self.loss_class = batch_opt(num_nodes, delay, rho=rho, det=det)
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

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        loss, predict = self.process_batch(input, real_val)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape, rmse, mae = self.get_metrics(predict, real_val)

        return loss.item(), mape, rmse, mae

    def get_metrics(self, predict, real_val):
        mape = util.masked_mape(predict, real_val, 0).item()
        rmse = util.masked_rmse(predict, real_val, 0).item()
        mae = util.masked_mae(predict, real_val, 0).item()

        return mape, rmse, mae

    def process_batch(self, input, real_val):
        b, d, f, n, t = input.shape
        input = input.reshape(b*d, f, n, t)

        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        mask = real_val != 0.0

        L_list = self.covariance.get_L()

        predict = predict.reshape(b, d, n, t)
        loss = self.loss(predict, real, L_list, mask).mean()
        return loss, predict.detach()

    def eval(self, input, real_val):
        self.model.eval()

        with torch.no_grad():
            loss, predict = self.process_batch(input, real_val)
            mape, rmse, mae = self.get_metrics(predict, real_val)
            return loss.item(), mape, rmse, mae
