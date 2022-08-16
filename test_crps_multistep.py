from matplotlib import animation
import sys
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
from Fixed_mdn_engine_multistep import MDN_trainer
# from Diag_Fixed_mdn_engine import MDN_trainer
import torch.nn as nn
import seaborn as sns
import properscoring as ps

os.chdir("/app/")
sys.argv = ['']
parser = argparse.ArgumentParser()

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
parser.add_argument("--mse_coef", type=float, default=0.1)

args = parser.parse_args()

# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiFalse
# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiTrue

# GWNMDN = "./logs_TRB_12sensors/GWN_MDN_20220620-200048_N10_R5_reg0.0_nhid4_pred12"
# GWNMDNreg = "./logs_TRB_12sensors/GWN_MDN_20220629-180633_N10_R5_reg0.1_nhid4_pred12"
# GWNMDNdiag = "./logs_TRB_12sensors/GWN_MDNdiag_20220619-145112_N10_R5_reg0.0_nhid4_pred12"
# flist = os.listdir("logs")

# GWNMDN = flist[0]
# GWNMDN = 'logs/GWNMDN_multistep_20220811-144913_N10_R5_reg0.0_nhid16_pred[2, 5, 8, 11]_rho0.01_diagFalse_msecoef1.0'
GWNMDN = "logs/GWNMDN_multistep_20220814-235047_N10_R5_reg0.0_nhid16_pred[2, 5, 8, 11]_rho0.01_diagFalse_msecoef1.0"
params = GWNMDN.split("_")[3:]

n_components = int(params[0].split("N")[1])
num_rank = int(params[1].split("R")[1])
reg_coef = float(params[2].split("reg")[1])
nhid = int(params[3].split("nhid")[1])
pred_len = [2, 5, 8, 11]
rho = float(params[5].split("rho")[1])
diag = True if params[6].split("diag")[1] == "True" else False
mse_coef = float(params[7].split("coef")[1])

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

dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, target_sensor_inds=target_sensor_inds, flow=False)
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
                     outlier_distribution=args.outlier_distribution, pred_len=args.pred_len, rho=args.rho, diag=args.diag,
                     mse_coef=args.mse_coef)


engine.load(model_path=args.model_path + '/model.pt',
            cov_path=args.model_path + '/covariance.pt',
            fc_w_path=args.model_path + '/fc_w.pt')


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
    L = info["scale_tril"][0]

    time_y = testy[:, 1, 0, -1]
    out_y = testy[:, 0, :, engine.pred_len]
    target = testy[:, 0, :, :]

    out_y = out_y.reshape(out_y.shape[0], -1)

    mu_out = mus[:, :]
    w_out = w[:, :]

    dist = engine.mdn_head.get_output_distribution(
        features={
            "w": w,
            "mu": mus,
            "scale_tril": L.unsqueeze(0).expand(out_y.shape[0], -1, -1, -1)
        }
    )

    # L = torch.tril(engine.covariance.L.unsqueeze(0))
    # mu0 = mus[:, 0, :]
    # mus[:, 1:, :] += mu0.unsqueeze(1).expand(-1, engine.n_components-1, -1)
    # L0 = L[:, 0, ...]
    # L[:, 1:, ...] += L0.unsqueeze(1).expand(-1, engine.n_components-1, -1, -1)

    # L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)] = torch.nn.functional.elu(
    #     L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)]) + 1

    for i in tqdm(range(time_y.shape[0])):
        time_y_i = time_y[i:(i+1)]
        out_y_i = out_y[i:(i+1)]
        mu_out_i = mu_out[i:(i+1), :]
        w_out_i = w_out[i:(i+1), :]
        # w_out_i = torch.softmax(w_out_i, -1)
        target_i = target[i:(i+1), :, :]

        engine.mdn_head.outlier_distribution = False
        dist_i = engine.mdn_head.get_output_distribution(features={
            'w': w_out_i,
            'mu': mu_out_i,
            'scale_tril': L.unsqueeze(0),
            'target': engine.scaler.transform(target_i)
        })

        sample_i = [dist_i.sample() for i in range(100)]
        # sample_i = torch.cat(sample_i, dim=0)
        sample_i = torch.stack(sample_i, dim=0)
        sample_i = sample_i[:, 0, :]
        sample_i = engine.scaler.inverse_transform(sample_i.cpu().numpy())
        # force positive sample_i
        sample_i[sample_i < 0] = 0

        for c in range(12):
            for t in range(4):
                score = ps.crps_ensemble(out_y_i[0, c * 4 + t].cpu(), sample_i[:, c * 4 + t])
                if score < 0:
                    print(i, c)
                mean_y_i = np.mean(sample_i[:, c * 4 + t])
                mu_out_i_scaled = engine.scaler.inverse_transform(mu_out_i[0, c * 4 + t].cpu().numpy())
                result.append([cnt, time_y_i.item(), c, t, out_y_i[0, c * 4 + t].cpu().item(),
                              np.log(score), mean_y_i, mu_out_i_scaled, w_out_i.numpy()])
        cnt += 1

    break

result_pd = pd.DataFrame(result, columns=['time', 'time_y', 'c', 't', 'out_y', 'score', 'sample_mean', 'mu', 'w'])
result_pd.to_csv('crps_result_MDNfull.csv', index=False)

result_pd["score_exp"] = np.exp(result_pd["score"])

cov_index_cmap = plt.cm.get_cmap('Set1', 10)


fig, ax = plt.subplots(figsize=(70, 40), nrows=4, ncols=12)

for sensor_id in range(12):
    y = [x[x["c"] == sensor_id]["out_y"] for x in [result_pd[result_pd["t"] == t] for t in range(4)]]
    mu = [x[x["c"] == sensor_id]["mu"] for x in [result_pd[result_pd["t"] == t] for t in range(4)]]
    w = [x[x["c"] == sensor_id]["w"] for x in [result_pd[result_pd["t"] == t] for t in range(4)]]
    score = [x[x["c"] == sensor_id]["score_exp"] for x in [result_pd[result_pd["t"] == t] for t in range(4)]]

    w = np.stack([np.concatenate(x.tolist()) for x in w])
    w_exp = np.exp(w)
    score = np.stack([x.tolist() for x in score])


    for t in range(4):
        ax[t, sensor_id].plot(y[t].reset_index()["out_y"], label="out_y")
        ax[t, sensor_id].plot(mu[t].reset_index()["mu"], label="mu")
        ax[t, sensor_id].legend()
        ax[t, sensor_id].set_title(f"sensor = {sensor_id} , t = {t}")
        ax[t, sensor_id].set_xlabel("time")
        ax[t, sensor_id].set_ylabel("value")
plt.show()
        # for c in range(args.n_components):
        #     ax[t, 1].plot(w_exp[t, :, c], label=f"w_{c}")
        # ax[t, 1].legend()
        # # set the same color for each component
        # ax[t, 1].set_prop_cycle(color=[cov_index_cmap(c) for c in range(args.n_components)])

        # ax[t, 2].plot(score[t, :], label=f"w_{t}")



# plot covariance matrix
# 10 covariance matrices at each subplot

fig2, axs2 = plt.subplots(figsize=(50, 20), nrows=2, ncols=5)

for i in range(10):
    cov_i = (engine.covariance.L[i] @ engine.covariance.L[i].T).cpu().detach().numpy()
    axs2[i // 5, i % 5].imshow(cov_i)
    axs2[i // 5, i % 5].set_title(f"{i}")
    # increase title size
    axs2[i // 5, i % 5].title.set_fontsize(40)
    # set title color
    axs2[i // 5, i % 5].title.set_color(cov_index_cmap(i))
fig2


fig3, axs3 = plt.subplots(figsize=(20, 50), nrows=5, ncols=2)

for i in range(5):
    cov_i = (engine.covariance.L[i] @ engine.covariance.L[i].T).cpu().detach().numpy()
    corr_i = np.corrcoef(cov_i)
    axs3[i,0].imshow(corr_i)
    axs3[i,0].set_title(f"corr_{i}")
    axs3[i,0].title.set_fontsize(40)
    axs3[i,0].title.set_color(cov_index_cmap(i))
    axs3[i,1].imshow(cov_i)
    axs3[i,1].set_title(f"cov_{i}")
    axs3[i,1].title.set_fontsize(40)
    axs3[i,1].title.set_color(cov_index_cmap(i))
plt.show()


sensor_id = 1
w = [x[x["c"] == sensor_id]["w"] for x in [result_pd[result_pd["t"] == t] for t in range(4)]]
w = np.stack([np.concatenate(x.tolist()) for x in w])
w_exp = np.exp(w)

fig, ax = plt.subplots(figsize = (30,20) , nrows = 1 , ncols = 1)
for c in range(args.n_components):
    ax.plot(w_exp[t, :, c], label=f"w_{c}")
ax.legend()
# increase legend size
plt.legend(fontsize=30)
plt.show()


fig3, axs3 = plt.subplots(figsize=(50, 20), nrows= 2, ncols=5)

for i in range(10):
    cov_i = (engine.covariance.L[i] @ engine.covariance.L[i].T).cpu().detach().numpy()
    corr_i = np.corrcoef(cov_i)
    axs3[i // 5, i % 5].imshow(corr_i)
    axs3[i // 5, i % 5].set_title(f"component_{i}")
    # increase title size
    axs3[i // 5, i % 5].title.set_fontsize(40)
plt.show()
