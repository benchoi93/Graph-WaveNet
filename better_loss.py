import torch
import torch.nn as nn
import math


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


def bessel_k_approx(z, v, max_iter=100):
    """Compute an approximation of the modified Bessel function of the second kind of order v using the asymptotic expansion"""
    result = torch.zeros_like(z)
    for k in range(max_iter):
        term = 1.0
        for i in range(k):
            term *= ((4 * v**2 - (2 * i + 1)**2) / (8 * z))
        add_term = term / math.factorial(k)
        result += add_term
    result *= torch.exp(-z) * torch.sqrt(math.pi / (2 * z))
    return result


class batch_opt(nn.Module):
    def __init__(self, num_nodes, delay, rho=1, det="mse", nll="MLD"):
        super(batch_opt, self).__init__()

        self.num_nodes = num_nodes
        self.delay = delay
        self.rho = rho
        self.det = det
        self.nll = nll

    def loss(self, pred, target, L_list, mask):

        if self.rho == 0:
            return self.masked_mse(pred, target, mask)
        elif self.rho == 1:
            return self.get_nll(pred, target, L_list)
        else:
            nll = self.get_nll(pred, target, L_list)
            mse = self.masked_mse(pred, target, mask)
            return self.rho * nll + (1-self.rho) * mse

    def get_nll(self, mu, target, L_list):
        if self.nll == "MGD":
            return self.get_nll_MGD(mu, target, L_list)
        elif self.nll == "MLD":
            return self.get_nll_MLD(mu, target, L_list)
        elif self.nll == "MLD_abs":
            return self.get_nll_MLD_abs(mu, target, L_list)
        elif self.nll == "GAL":
            return self.get_nll_GAL(mu, target, L_list)

    def get_nll_GAL(self, pred, target, L_list):
        b, d, n, t = pred.shape
        t = t // 2
        dnt = d * n * t

        mu = pred[..., :t]
        gamma = pred[..., t:]

        mu = mu.unsqueeze(1)
        gamma = gamma.unsqueeze(1)
        R_ext = (mu - target)
        gamma_ext = (gamma - target)

        L_list = [l.transpose(-1, -2).unsqueeze(0).repeat(b, 1, 1, 1) for l in L_list]
        logdet = [l.diagonal(dim1=-1, dim2=-2).log().sum(-1) for l in L_list]

        L_x = kron_vec_prod(L_list, R_ext, align=2)
        L_gamma = kron_vec_prod(L_list, gamma_ext, align=2)

        logdet = sum([dnt*ll/L_list[i].shape[-1] for i, ll in enumerate(logdet)])

        nll = -(
            logdet +
            (- 0.5 * torch.log(2 + L_gamma.pow(2).sum((-1, -2, -3)))) +
            (- torch.sqrt((2 + L_gamma.pow(2).sum((-1, -2, -3)))) * L_x.pow(2).sum((-1, -2, -3))) +
            (L_x * L_gamma).sum((-1,-2,-3))
        )
        # nll = - torch.logsumexp(nll, dim=1)
        return nll

    def get_nll_MLD(self, mu, target, L_list):
        b, d, n, t = mu.shape
        dnt = d * n * t

        mu = mu.unsqueeze(1)
        R_ext = (mu - target)

        L_list = [l.transpose(-1, -2).unsqueeze(0).repeat(b, 1, 1, 1) for l in L_list]
        logdet = [l.diagonal(dim1=-1, dim2=-2).log().sum(-1) for l in L_list]

        L_x = kron_vec_prod(L_list, R_ext, align=2)
        mahabolis = L_x.norm(p=2, dim=(-1, -2, -3))

        logdet = sum([dnt*ll/L_list[i].shape[-1] for i, ll in enumerate(logdet)])
        # nll = logdet + (1-dnt)/2 * torch.log(mahabolis) - math.sqrt(2) * mahabolis
        nll = -(logdet - math.sqrt(2) * mahabolis)

        # nll = - torch.logsumexp(nll, dim=1)
        return nll

    def get_nll_MLD_abs(self, mu, target, L_list):
        b, d, n, t = mu.shape
        dnt = d * n * t

        mu = mu.unsqueeze(1)
        R_ext = (mu - target)

        L_list = [l.transpose(-1, -2).unsqueeze(0).repeat(b, 1, 1, 1) for l in L_list]
        logdet = [l.diagonal(dim1=-1, dim2=-2).log().sum(-1) for l in L_list]

        L_x = kron_vec_prod(L_list, R_ext, align=2)
        mahabolis = L_x.norm(p=1, dim=(-1, -2, -3))

        logdet = sum([dnt*ll/L_list[i].shape[-1] for i, ll in enumerate(logdet)])
        nll = -(logdet - math.sqrt(2) * mahabolis)

        # nll = - torch.logsumexp(nll, dim=1)
        return nll

    def get_nll_MGD(self, mu, target, L_list):
        b, d, n, t = mu.shape

        mu = mu.unsqueeze(1)

        # mask = (target != 0)
        R_ext = (mu - target)
        # R_flatten = R_ext.reshape(b, d * n * t).unsqueeze(1)
        # R_ext = R_ext.unsqueeze(1)

        L_list = [l.transpose(-1, -2).unsqueeze(0).repeat(b, 1, 1, 1) for l in L_list]
        logdet = [l.diagonal(dim1=-1, dim2=-2).log().sum(-1) for l in L_list]

        L_x = kron_vec_prod(L_list, R_ext, align=2)
        mahabolis = L_x.pow(2).sum((-1, -2, -3))

        dnt = d * n * t
        logdet = sum([dnt*ll/L_list[i].shape[-1] for i, ll in enumerate(logdet)])

        nll = -(-dnt/2 * math.log(2*math.pi) - 0.5 * mahabolis + logdet)

        # nll = - torch.logsumexp(nll, dim=1)
        return nll

    def masked_mse(self, pred, target, mask):
        if self.nll == "GAL":
            b, d, n, t = pred.shape
            t = t // 2
            pred = pred[..., :t]

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
