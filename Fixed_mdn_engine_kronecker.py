import torch.optim as optim
from model import *
import util
import math
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
    def __init__(self, n_components, n_vars, num_pred, rho=0.5, diag=False, trainL=True):
        super(FixedMDN, self).__init__()
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

        reg_loss = self.get_sparsity_regularization_loss(dist)

        target = y[:, :, self.pred_len - 1]
        # min_mse, _ = ((dist.mean - target)**2).min(1)
        # min_mse, _ = ((features["mu"] - target.unsqueeze(1)) ** 2).min(1)
        mse_loss = ((dist.mean - target)**2).mean(1).mean()

        # loss = nll_loss + reg_loss * self.reg_coef + mse_loss*100
        loss = features["rho"] * nll_loss + reg_loss * self.reg_coef
        # loss = mse_loss

        return loss, nll_loss.item(), reg_loss.item(), mse_loss.item()

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
                 outlier_distribution_kwargs=None, mse_coef=0.1, rho=0.1):

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

    def forward(self, features):
        # input : features = dict of tensors
        #    - features['w'] : (batch_size, n_components)
        #    - features['mu'] : (batch_size, n_components, n_vars)
        #    - features['D'] : (batch_size, n_components, n_vars)
        #    - features['V'] : (batch_size, n_components, n_vars, n_rank)
        y = features['target']
        target = y[:, :, self.pred_len]
        # features['target'] = target

        # dist = self.get_output_distribution(features)
        target = y[:, :, self.pred_len].reshape(y.shape[0], -1)
        # nll_loss_torch = - dist.log_prob(target).mean()
        # nll_loss = self.get_nll(features, target).mean()
        nll_loss = self.get_nll_mnd(features, target).mean()

        reg_loss = 0
        # reg_loss = self.get_sparsity_regularization_loss(features)
        # dist = self.get_output_distribution(features, consider_neighbors=False)

        # min_mse, _ = ((dist.mean - target)**2).min(1)
        # min_mse, _ = ((features["mu"] - target.unsqueeze(1)) ** 2).min(1)
        mse_loss = ((features["mu"] - target)**2).mean()

        loss = self.rho * nll_loss + reg_loss * self.reg_coef + self.mse_coef * mse_loss
        # loss = mse_loss

        return loss, nll_loss.item(), reg_loss, mse_loss.item()

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

    def sample(self, features, n=None):
        # dist = self.get_output_distribution(features)
        ws,wt, mu, L_s, L_t = self.get_parameters(features)

        U_inv, V_inv = self.get_mnd(features)
        B, C = ws.shape
        N = self.n_vars
        T = len(self.pred_len)

        U = torch.inverse(U_inv)
        V = torch.inverse(V_inv)
        mu = mu.reshape(B, N, T)

        device = mu.device
        iid_dist = Dist.Independent(Dist.Normal(torch.zeros((B, N, T), device=device), torch.ones((B, N, T), device=device)), 1)

        if n is None:
            samples = iid_dist.sample()
        else:
            samples = iid_dist.sample((n,))

        samples = mu.unsqueeze(0) + torch.einsum("bln,sbnt, btk->sblk", U, samples, V)

        return samples
        # torch.einsum("ln,nt,tk->lk", U[0],samples[0,0],V[0])

    def get_mnd(self, features):
        ws, wt, mu, L_s, L_t = self.get_parameters(features)

        b = mu.shape[0]
        c = self.n_components
        d = self.n_vars
        t = len(self.pred_len)

        prc_spatial = torch.einsum("ijk,ikl->ijl", L_s, L_s.transpose(-1, -2))
        prc_temporal = torch.einsum("ijk,ikl->ijl", L_t, L_t.transpose(-1, -2))

        U_inv = torch.zeros((b, d, d), device=prc_spatial.device)
        for i in range(c):
            U_inv += ws[:, i].unsqueeze(-1).unsqueeze(-1) * prc_spatial[i]

        V_inv = torch.zeros((b, t, t), device=prc_temporal.device)
        for i in range(c):
            V_inv += wt[:, i].unsqueeze(-1).unsqueeze(-1) * prc_temporal[i]

        return U_inv, V_inv

    def get_nll_mnd(self, features, target):
        ws, wt, mu, L_s, L_t = self.get_parameters(features)

        U_inv, V_inv = self.get_mnd(features)
        # prc = util.kron(U_inv, V_inv)

        B, C = ws.shape
        N = self.n_vars
        T = len(self.pred_len)

        X = target.reshape(B, N, T)
        mu = mu.reshape(B, N, T)
        Z = X - mu

        mahabolis = -0.5 * torch.einsum("bij,bjk,bkl,blp->bip", V_inv, Z.transpose(-1, -2), U_inv, Z)
        mahabolis = mahabolis.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

        # L_s_diag = torch.diagonal(L_s, dim1=1, dim2=2)
        # L_t_diag = torch.diagonal(L_t, dim1=1, dim2=2)

        # L_s_diag = torch.repeat_interleave(L_s_diag, T, dim=1)
        # L_t_diag = L_t_diag.repeat((1, N))
        # ws.unsqueeze(-1) * L_s_diag.unsqueeze(0)
        # L_s_diag
        # L_kron_diag = L_s_diag * L_t_diag

        Ulogdet = torch.logdet(U_inv)
        Vlogdet = torch.logdet(V_inv)
        # Ulogdet = - T / 2 * (ws.unsqueeze(-1) * L_s_diag.unsqueeze(0)).sum(1).log().sum(1)
        # Vlogdet = - N / 2 * (wt.unsqueeze(-1) * L_t_diag.unsqueeze(0)).sum(1).log().sum(1)

        # half_log_det = (w.unsqueeze(-1) * L_kron_diag.unsqueeze(0)).sum(1).log().sum(1)

        nll = -(-N*T/2 * math.log(2*math.pi) + mahabolis + N/2 * Vlogdet + T/2 * Ulogdet)

        # dist = self.get_output_distribution(features)
        # nll_loss_torch = - dist.log_prob(target)

        return nll

    def get_nll(self, features, target):
        w, mu, L_s, L_t = self.get_parameters(features)

        z = target - mu
        N = self.n_vars
        T = len(self.pred_len)

        assert(N * T == z.shape[1])

        cov_s = torch.einsum("bij, bjk -> bik", L_s, L_s.transpose(-1, -2))
        cov_t = torch.einsum("bij, bjk -> bik", L_t, L_t.transpose(-1, -2))

        prc_s = torch.inverse(cov_s)
        prc_t = torch.inverse(cov_t)

        # now = time.time()
        # M = 0
        # for i in range(N):
        #     for j in range(N):
        #         y_front = z[:, i*T:(i+1)*T]
        #         y_back = z[:, j*T:(j+1)*T]

        #         # for k in range(cov_t.shape[0]):
        #         K_s = prc_s[:, i, j].unsqueeze(-1).unsqueeze(-1)
        #         K_t = prc_t
        #         K = K_s * K_t

        #         K_final = w.unsqueeze(-1).unsqueeze(-1) * K.unsqueeze(0).expand(z.shape[0], -1, -1, -1)
        #         K_final = K_final.sum(1)

        #         M += torch.einsum("bij,bjk,bkl->bil", y_front.unsqueeze(1), K_final, y_back.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        # prc_time = time.time() - now

        M = 0
        for i in range(T):
            for j in range(T):
                y_front = z[:, i::T]
                y_back = z[:, j::T]

                K_s = prc_s
                K_t = prc_t[:, i, j].unsqueeze(-1).unsqueeze(-1)

                K_final = w.unsqueeze(-1).unsqueeze(-1) * (K_s * K_t).unsqueeze(0).expand(z.shape[0], -1, -1, -1)
                K_final = K_final.sum(1)

                M += torch.einsum("bij,bjk,bkl->bil", y_front.unsqueeze(1), K_final, y_back.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        # determinants

        L_s_diag = torch.diagonal(L_s, dim1=1, dim2=2)
        L_t_diag = torch.diagonal(L_t, dim1=1, dim2=2)

        L_s_diag = torch.repeat_interleave(L_s_diag, T, dim=1)
        L_t_diag = L_t_diag.repeat((1, N))

        L_kron_diag = L_s_diag * L_t_diag

        half_log_det = (w.unsqueeze(-1) * L_kron_diag.unsqueeze(0)).sum(1).log().sum(1)

        nll_loss = -0.5 * (N*T * math.log(2 * math.pi) + M) - half_log_det

        # dist = self.get_output_distribution(features)
        # nll_loss_torch = - dist.log_prob(target)

        # diff = nll_loss + nll_loss_torch

        return - nll_loss

    def get_parameters(self, features):
        # input : features : dict of tensors, keys: w, mu, D, V
        # check if 'w' , 'mu' , 'L' are in features.keys()
        # raise error if not
        # assert('w' in features.keys())
        # assert('mu' in features.keys())
        # assert('cov' in features.keys())
        # return features['w'], features['mu'], features['cov']
        return features['ws'], features['wt'], features['mu'], features['L_spatial'], features['L_temporal']


class MDN_trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, num_rank, nhid, dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, n_components, reg_coef,
                 mode="cholesky", time_varying=False, consider_neighbors=False, outlier_distribution=True, pred_len=12, rho=0.5, diag=False, mse_coef=0.1, nonlinearity="softmax"):

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
                                        mse_coef=mse_coef)
        # dim_w = [batn_components]
        # dims_c = dim_w, dim_mu, dim_U_entries, dim_i
        # self.mdn = FrEIA.modules.GaussianMixtureModel(dims_in, dims_c)
        self.covariance = FixedMDN(n_components, num_nodes, self.num_pred, rho=rho, diag=diag, trainL=rho != 0)

        self.fc_ws = nn.Sequential(
            nn.Linear(self.n_components*num_nodes*self.out_per_comp, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, self.n_components)
        )

        self.fc_wt = nn.Sequential(
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
        self.fc_ws.to(device)
        self.fc_wt.to(device)

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.fc_ws.parameters()) + list(self.fc_wt.parameters()) +
                                    list(self.covariance.parameters()), lr=lrate, weight_decay=wdecay)
        # self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

        import datetime
        # self.logdir = f'./logs/GWN_MDN_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_N{n_components}_R{num_rank}_reg{reg_coef}_nhid{nhid}_nei{consider_neighbors}'
        self.logdir = f'./logs/GWNMDN_kronecker_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_N{n_components}_R{num_nodes}_reg{reg_coef}_nhid{nhid}_pred{pred_len}_rho{rho}_diag{diag}_msecoef{mse_coef}_nlin{nonlinearity}'

        self.summary = SummaryWriter(logdir=f'{self.logdir}')
        self.cnt = 0
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-1)

        if nonlinearity == "softmax":
            self.w_act = nn.Softmax(dim=-1)
        elif nonlinearity == "softplus":
            self.w_act = nn.Softplus()
        elif nonlinearity == "elu":
            def elusplusone(x): return nn.ELU()(x) + 1
            self.w_act = elusplusone

    def save(self, best=False):
        if best:
            torch.save(self.model.state_dict(), f'{self.logdir}/best_model.pt')
            torch.save(self.covariance.state_dict(), f'{self.logdir}/best_covariance.pt')
            torch.save(self.fc_ws.state_dict(), f'{self.logdir}/best_fc_ws.pt')
            torch.save(self.fc_wt.state_dict(), f'{self.logdir}/best_fc_wt.pt')

        torch.save(self.model.state_dict(), f'{self.logdir}/model.pt')
        torch.save(self.covariance.state_dict(), f'{self.logdir}/covariance.pt')
        torch.save(self.fc_ws.state_dict(), f'{self.logdir}/fc_ws.pt')
        torch.save(self.fc_wt.state_dict(), f'{self.logdir}/fc_wt.pt')

    def load(self, model_path, cov_path, fc_ws_path, fc_wt_path):
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.covariance.load_state_dict(torch.load(cov_path), strict=True)
        self.fc_ws.load_state_dict(torch.load(fc_ws_path), strict=True)
        self.fc_wt.load_state_dict(torch.load(fc_wt_path), strict=True)

    def train(self, input, real_val, eval=False):
        if not eval:
            self.mdn_head.training = True
            self.model.train()
            self.fc_ws.train()
            self.fc_wt.train()
            self.covariance.train()

        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)

        # output = output.view(-1, self.num_nodes, self.n_components, self.out_per_comp)
        mus = output[:, 0, :, :self.num_pred]
        mus = mus.reshape(-1, self.num_pred * self.num_nodes)

        output = output.reshape(-1, self.n_components*self.num_nodes * self.out_per_comp)
        ws = self.fc_ws(output)
        ws = self.w_act(ws)

        wt = self.fc_wt(output)
        wt = self.w_act(wt)

        L_spatial = self.covariance.L1
        L_temporal = self.covariance.L2
        # L_spatial = L_spatial.reshape(self.n_components * self.n_components, self.num_nodes, self.num_nodes)
        # L_temporal = L_temporal.transpose(0, 1)
        # L_temporal = L_temporal.reshape(self.n_components * self.n_components, self.num_pred, self.num_pred)

        L_spatial[:,  torch.arange(L_spatial.shape[-1]), torch.arange(L_spatial.shape[-1])] = torch.nn.functional.elu(
            L_spatial[:,  torch.arange(L_spatial.shape[-1]), torch.arange(L_spatial.shape[-1])]) + 1
        L_temporal[:,  torch.arange(L_temporal.shape[-1]), torch.arange(L_temporal.shape[-1])] = torch.nn.functional.elu(
            L_temporal[:,  torch.arange(L_temporal.shape[-1]), torch.arange(L_temporal.shape[-1])]) + 1

        L_temporal[:, 0, 0] = 1

        # cov = torch.zeros(output.shape[0] , self.n_components * self.n_components , self.num_pred * self.num_nodes , self.num_pred * self.num_nodes)
        # cov_spatial = torch.einsum("ijk,ikl->ijl", L_spatial, L_spatial.transpose(-1, -2))
        # cov_temporal = torch.einsum("ijk,ikl->ijl", L_temporal, L_temporal.transpose(-1, -2))

        # cov = util.kron(cov_spatial, cov_temporal)
        # cov = cov.unsqueeze(0).expand(output.shape[0], -1, -1, -1)
        # cov = (w.unsqueeze(-1).unsqueeze(-1).expand_as(cov) * cov).sum(1)

        # b, c, d, _ = cov.size()
        # cov_new = torch.zeros((b, d, d), device=cov.device)

        # for i in range(c):
        # cov_new += w[:, i].unsqueeze(-1).unsqueeze(-1) * cov[:, i, :, :]
        scaled_real_val = self.scaler.transform(real_val)

        features = {'ws': ws,
                    'wt': wt,
                    'mu': mus,
                    'L_spatial': L_spatial,
                    "L_temporal": L_temporal,
                    # "cov": cov_new,
                    "rho": self.covariance.rho,
                    'target': scaled_real_val}

        loss, nll_loss, reg_loss, mse_loss = self.mdn_head.forward(
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
        real = real_val[:, :, self.pred_len].reshape(real_val.shape[0], -1)

        # output = self.mdn_head.sample(features={'w': w, 'mu': mus, 'scale_tril': L})
        # output = self.mdn_head.get_output_distribution(features).mean
        output = mus
        predict = self.scaler.inverse_transform(output)
        predict[predict < 0] = 0
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        # crps = self.specific_eval(features={'w': w, 'mu': mus, 'scale_tril': L, "rho": self.covariance.rho, 'target': scaled_real_val})

        info = {
            "ws": ws,
            "wt": wt,
            "mu": mus,
            "L_spatial": L_spatial,
            "L_temporal": L_temporal,
            # "cov": cov_new,
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
        self.fc_ws.eval()
        self.fc_wt.eval()
        self.covariance.eval()

        with torch.no_grad():
            info = self.train(input, real_val, eval=True)

        crps, ES = self.specific_eval(features=info)
        # crps = 0

        info['crps'] = crps
        info["ES"] = ES.cpu().numpy()

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
        output = self.mdn_head.sample(features=features, n=100)
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

    def plot_cov(self, features):
        # dist = self.mdn_head.get_output_distribution(features)
        # sample_cov = dist.component_distribution.covariance_matrix[0]
        # sample_prec = dist.component_distribution.precision_matrix[0]
        L_spatial = self.covariance.L1
        L_temporal = self.covariance.L2

        L_spatial[:,  torch.arange(L_spatial.shape[-1]), torch.arange(L_spatial.shape[-1])] = torch.nn.functional.elu(
            L_spatial[:,  torch.arange(L_spatial.shape[-1]), torch.arange(L_spatial.shape[-1])]) + 1
        L_temporal[:,  torch.arange(L_temporal.shape[-1]), torch.arange(L_temporal.shape[-1])] = torch.nn.functional.elu(
            L_temporal[:,  torch.arange(L_temporal.shape[-1]), torch.arange(L_temporal.shape[-1])]) + 1

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
