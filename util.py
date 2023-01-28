import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import matplotlib
import matplotlib.pyplot as plt


def universal_fig(figsize=(3, 3), fontsize=12, axislinewidth=1, markersize=5, text=None, limits=[-7, 7], offset=[-44, 12], projection=None, fontfamily=["Helvetica", "Arial"], contain_latex=False):
    '''
    Create universal figure settings with publication quality
    returen fig, ax (similar to plt.plot)
    fig, ax = universal_fig()
    '''
    # ----------------------------------------------------------------
    if projection is None:
        fig, ax = plt.subplots(frameon=False)
    else:
        fig, ax = plt.subplots(frameon=False, subplot_kw=dict(projection=projection))
    fig.set_size_inches(figsize)
    matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": fontfamily, "size": fontsize})
    matplotlib.rc('pdf', fonttype=42, use14corefonts=True, compression=6)
    matplotlib.rc('ps', useafm=True, usedistiller='none', fonttype=42)
    matplotlib.rc("axes", unicode_minus=False, linewidth=axislinewidth, labelsize='medium')
    matplotlib.rc("axes.formatter", limits=limits)
    matplotlib.rc('savefig', bbox='tight', format='eps', frameon=False, pad_inches=0.05)
    matplotlib.rc('legend')
    matplotlib.rc('lines', marker=None, markersize=markersize)
    matplotlib.rc('text', usetex=False)
    matplotlib.rc('xtick', direction='in')
    matplotlib.rc('xtick.major', size=4)
    matplotlib.rc('xtick.minor', size=2)
    matplotlib.rc('ytick', direction='in')
    matplotlib.rc('lines', linewidth=1)
    matplotlib.rc('ytick.major', size=4)
    matplotlib.rc('ytick.minor', size=2)
    matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
    matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
    matplotlib.rc('mathtext', fontset='stixsans')

    if contain_latex:
        matplotlib.rc('ps', useafm=False, usedistiller='none', fonttype=3)
        matplotlib.rc('pdf', fonttype=3, use14corefonts=True, compression=6)

    matplotlib.rc('legend', fontsize='medium', frameon=False,
                  handleheight=0.5, handlelength=1, handletextpad=0.4, numpoints=1)
    if text is not None:
        w = ax.annotate(text, xy=(0, 1), xycoords='axes fraction', fontsize='large', weight='bold',
                        xytext=(offset[0]/12*fontsize, offset[1]/12*fontsize), textcoords='offset points', ha='left', va='top')
        print(w.get_fontname())
    # ----------------------------------------------------------------
    # end universal settings
    return fig, ax


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self, force_idx=None):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]

                if force_idx is not None:
                    x_i[0] = self.xs[force_idx, ...]
                    y_i[0] = self.ys[force_idx, ...]

                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, target_sensor_inds=None, flow=True):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        if target_sensor_inds is not None:
            if dataset_dir == "data/PEMS-BAY-2022":
                if flow:
                    data['x_' + category] = cat_data['x'][:, :, target_sensor_inds, :][:, :, :, (0, 2)]
                    data['y_' + category] = cat_data['y'][:, :, target_sensor_inds, :][:, :, :, (0, 2)]
                else:
                    # speed
                    data['x_' + category] = cat_data['x'][:, :, target_sensor_inds, :][:, :, :, (1, 2)]
                    data['y_' + category] = cat_data['y'][:, :, target_sensor_inds, :][:, :, :, (1, 2)]

            else:
                data['x_' + category] = cat_data['x'][:, :, target_sensor_inds, :]
                data['y_' + category] = cat_data['y'][:, :, target_sensor_inds, :]
        else:
            data['x_' + category] = cat_data['x']
            data['y_' + category] = cat_data['y']

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, pad_with_last_sample=False)
    data['scaler'] = scaler
    return data


def masked_mse(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def get_missing_rate(data, nanvalue=0):
    num_missing = (data['x'][:, :, :, 0] == nanvalue).sum()
    num_total = data['x'][:, :, :, 0].size

    return num_missing / num_total * 100


# metr = np.load("/app/data/METR-LA/train.npz")
# pems = np.load("/app/data/PEMS-BAY/train.npz")
# pems2022 = np.load("/app/data/PEMS-BAY-2022/train.npz")

# print(get_missing_rate(metr))
# print(get_missing_rate(pems))
# print(get_missing_rate(pems2022))

# (metr['x'][:, :, :, 0] == 0)

# np.argsort((pems2022['x'][:, :, :, 0] == 0).sum((0, 1)))
# np.sort((metr['x'][:, :, :, 0] == 0).sum((0, 1)))


# ((pems2022['x'][:, :, list(np.argsort((pems2022['x'][:, :, :, 0] == 0).sum((0, 1)))[:-5]), 0] == 0).sum() /
#  pems2022['x'][:, :, list(np.argsort((pems2022['x'][:, :, :, 0] == 0).sum((0, 1)))[:-5]), 0].size)*100


# np.histogram((pems['x'][:, :, :, 0] == 0).sum((0, 1)), bins=100)
# np.histogram((pems2022['x'][:, :, :, 0] == 0).sum((0, 1)), bins=100)
# np.histogram((metr['x'][:, :, :, 0] == 0).sum((0, 1)), bins=100)
