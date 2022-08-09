import pandas as pd
import os
from tqdm import tqdm
import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
# from mdn_engine import MDN_trainer
from Fixed_mdn_engine import MDN_trainer
# from Diag_Fixed_mdn_engine import MDN_trainer
import torch.nn as nn
import seaborn as sns
import properscoring as ps


import sys
sys.argv = ['']
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='')
parser.add_argument('--data', type=str, default='data/PEMS-BAY-2022', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx_bay.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--num-rank', type=int, default=5, help='')
parser.add_argument('--nhid', type=int, default=16, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=1000, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save', type=str, default='./garage/pems', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--n_components', type=int, default=10, help='experiment id')
parser.add_argument('--reg_coef', type=float, default=0.1, help='experiment id')
parser.add_argument('--save_every', type=int, default=20, help='experiment id')
parser.add_argument("--consider_neighbors", action="store_true", help="consider neighbors")
parser.add_argument("--outlier_distribution", action="store_true", help="outlier_distribution")
parser.add_argument("--pred-len", type=int, default=12)
parser.add_argument("--rho", type=float, default=0.1)
parser.add_argument("--diag", action="store_true")

args = parser.parse_args()

# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiFalse
# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiTrue

# GWNMDN = "./logs_TRB_12sensors/GWN_MDN_20220620-200048_N10_R5_reg0.0_nhid4_pred12"
# GWNMDNreg = "./logs_TRB_12sensors/GWN_MDN_20220629-180633_N10_R5_reg0.1_nhid4_pred12"
# GWNMDNdiag = "./logs_TRB_12sensors/GWN_MDNdiag_20220619-145112_N10_R5_reg0.0_nhid4_pred12"
flist = os.listdir("logs")

# GWNMDN = flist[0]
GWNMDN = 'logs/GWN_MDN_20220808-182736_N10_R5_reg3.0_nhid16_pred24_rho0.1_diagFalse_mse_coef100.0'
params = GWNMDN.split("_")[3:]

n_components = int(params[0].split("N")[1])
num_rank = int(params[1].split("R")[1])
reg_coef = float(params[2].split("reg")[1])
nhid = int(params[3].split("nhid")[1])
pred_len = int(params[4].split("pred")[1])
rho = float(params[5].split("rho")[1])
diag = bool(params[6].split("diag")[1])
mse_coef = float(params[8].split("coef")[1])

args.n_components = n_components
args.nhid = nhid
args.num_rank = num_rank
args.reg_coef = reg_coef
args.pred_len = pred_len
args.rho = rho
args.model_path = GWNMDN
args.diag = diag
args.mse_coef = mse_coef


# def main():
# set seed
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# load data
device = torch.device(args.device)
sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)

target_sensors = ['404444',
                  '400582',
                  '400097',
                  '401224',
                  '400828',
                  '400648',
                  '404434',
                  '400222',
                  '400952',
                  '401210',
                  '400507',
                  '400185'
                  ]

target_sensor_inds = [sensor_id_to_ind[i] for i in target_sensors]

dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, target_sensor_inds=target_sensor_inds)
scaler = dataloader['scaler']
supports = [torch.tensor(i).to(device) for i in adj_mx]

print(args)

if args.randomadj:
    adjinit = None
else:
    adjinit = supports[0]

if args.aptonly:
    supports = None

adjinit = adjinit[:, target_sensor_inds][target_sensor_inds, :]

# engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
#                  args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
#                  adjinit)

engine = MDN_trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.num_rank, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit, n_components=args.n_components, reg_coef=args.reg_coef, consider_neighbors=args.consider_neighbors,
                     outlier_distribution=args.outlier_distribution, pred_len=args.pred_len, rho=args.rho, diag=args.diag)


engine.load(model_path=args.model_path + '/best_model.pt',
            cov_path=args.model_path + '/best_covariance.pt',
            fc_w_path=args.model_path + '/best_fc_w.pt')


result = []
cnt = 0

for iter, (x, y) in tqdm(enumerate(dataloader['test_loader'].get_iterator())):
    with torch.no_grad():
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        info = engine.eval(testx, testy[:, 0, :, :])

    w = info["w"]
    mus = info["mu"]
    L = info["scale_tril"]

    time_y = testy[:, 1, 0, -1]
    out_y = testy[:, 0, :, engine.pred_len - 1]
    target = testy[:, 0, :, :]

    mu_out = mus[:, :, :]
    w_out = w[:, :]

    L = torch.tril(engine.covariance.L.unsqueeze(0))
    mu0 = mus[:, 0, :]
    mus[:, 1:, :] += mu0.unsqueeze(1).expand(-1, engine.n_components-1, -1)
    L0 = L[:, 0, ...]
    L[:, 1:, ...] += L0.unsqueeze(1).expand(-1, engine.n_components-1, -1, -1)

    L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)] = torch.nn.functional.elu(
        L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)]) + 1

    for i in tqdm(range(time_y.shape[0])):
        time_y_i = time_y[i:(i+1)]
        out_y_i = out_y[i:(i+1)]
        mu_out_i = mu_out[i:(i+1), :]
        w_out_i = w_out[i:(i+1), :]
        w_out_i = torch.softmax(w_out_i, -1)
        target_i = target[i:(i+1), :, :]

        engine.mdn_head.outlier_distribution = False
        dist_i = engine.mdn_head.get_output_distribution(features={
            'w': w_out_i,
            'mu': mu_out_i,
            'scale_tril': L,
            'target': engine.scaler.transform(target_i)
        })
        dist_i = torch.distributions.MultivariateNormal(
            loc=mu_out_i,
            scale_tril=L
        )

        sample_i = [dist_i.sample()[0] for i in range(100)]
        sample_i = torch.cat(sample_i, dim=0)
        sample_i = engine.scaler.inverse_transform(sample_i.cpu().numpy())
        # force positive sample_i
        sample_i[sample_i < 0] = 0

        for c in range(12):
            score = ps.crps_ensemble(out_y_i[0, c], sample_i[:, c])
            if score < 0:
                print(i, c)
                break
            # score = 0
            mean_y_i = np.mean(sample_i[:, c])
            mu_out_i_scaled = engine.scaler.inverse_transform(mu_out_i[0, :, c].cpu().numpy())
            result.append([cnt, time_y_i.item(), c, out_y_i[0, c].item(), np.log(score), mean_y_i, mu_out_i_scaled, w_out_i])
        cnt += 1

    break

result_pd = pd.DataFrame(result, columns=['time', 'time_y', 'c', 'out_y', 'score', 'sample_mean', 'mu', 'w'])
result_pd.to_csv('crps_result_MDNfull.csv', index=False)
# result_pd.to_csv('crps_result_MDN_reg0.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)

result_pd["score_exp"] = result_pd["score"].apply(lambda x: np.exp(x))
# sns.set_theme(style="darkgrid")
sns.heatmap(
    result_pd.pivot(index="time", columns="c", values="score"), cmap="BuGn"
)
plt.show()


fig, ax = plt.subplots(figsize=(20, 10), nrows=1, ncols=2)
plt.plot(result_pd["time_y"], result_pd["out_y"], ".", label="true")
plt.show()


sns.heatmap(
    result_pd.pivot(index="time", columns="c", values="score_exp"), cmap="BuGn"
)
plt.show()


np.exp(result_pd["score"].to_numpy()).mean()


fig, ax = plt.subplots(figsize=(30, 10), nrows=1, ncols=3)

plt.cla()
sensor_id = 2

mus = np.stack(result_pd[result_pd["c"] == sensor_id]["mu"].tolist())
ws = np.stack(result_pd[result_pd["c"] == sensor_id]["w"].tolist())
ax[0].plot((result_pd[result_pd["c"] == sensor_id]["out_y"]).reset_index()["out_y"])
ax[0].plot((result_pd[result_pd["c"] == sensor_id]["sample_mean"]).reset_index()["sample_mean"], color="red")
for i in range(mus.shape[1]):
    ax[1].plot(mus[:, i], label=str(i))
# for i in range(mus.shape[1]):
#     ax[2].plot(ws[:, 0, i], label=str(i))
ax[2].plot(result_pd[result_pd["c"] == sensor_id]['score_exp'])
ax[0].set_title("out_y")
ax[1].set_title("mu")
ax[2].set_title("w")
fig


# result_pd_ind = pd.read_csv("crps_result_MDNind.csv")
result_pd_diag = pd.read_csv("crps_result_MDNdiag.csv")
result_pd_fullcov = pd.read_csv("crps_result_MDNfull.csv")
result_pd_diag["score_exp"] = result_pd_diag["score"].apply(lambda x: np.exp(x))
result_pd_fullcov["score_exp"] = result_pd_fullcov["score"].apply(lambda x: np.exp(x))


fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
plt.scatter(result_pd_diag["score"], result_pd_fullcov["score"], alpha=0.1)
# draw x=y line
plt.plot([0, 5], [0, 5], 'k--')
plt.xlabel("MDN ind cov")
plt.ylabel("MDN full cov")
plt.xlim([-1, 5])
plt.ylim([-1, 5])
plt.show()

# result_pd_diag['score'] = np.exp(result_pd_diag['score'])
# result_pd_fullcov['score'] = np.exp(result_pd_fullcov['score'])

result_pd_diag['score_exp'].mean()
result_pd_fullcov['score_exp'].mean()

# sns.set_theme(style="darkgrid")
sns.heatmap(
    result_pd_diag.pivot(index="time", columns="c", values="score"), cmap="BuGn"
)
plt.show()

# sns.set_theme(style="darkgrid")
sns.heatmap(
    result_pd_fullcov.pivot(index="time", columns="c", values="score"), cmap="BuGn"
)
plt.show()

sns.heatmap(
    result_pd_diag.pivot(index="time", columns="c", values="score_exp"), cmap="BuGn"
)
plt.show()

# sns.set_theme(style="darkgrid")
sns.heatmap(
    result_pd_fullcov.pivot(index="time", columns="c", values="score_exp"), cmap="BuGn"
)
plt.show()
