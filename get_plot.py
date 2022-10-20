
from Fixed_mdn_engine_residual import MDN_trainer as trainer_residual
from Fixed_mdn_engine_multistep import MDN_trainer as trainer_multistep
import pandas as pd
import os
import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
# from mdn_engine import MDN_trainer
from Fixed_mdn_engine_residual import MDN_trainer
# from Diag_Fixed_mdn_engine import MDN_trainer

import sys

sys.argv = [""]
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/PEMS-BAY-2022', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx_bay.pkl', help='adj data path')
# parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
# parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--num-rank', type=int, default=5, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=300, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./garage/pems', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--n_components', type=int, default=1, help='experiment id')
parser.add_argument('--reg_coef', type=float, default=0.1, help='experiment id')
parser.add_argument('--save_every', type=int, default=20, help='experiment id')
parser.add_argument("--consider_neighbors", action="store_true", help="consider neighbors")
parser.add_argument("--outlier_distribution", action="store_true", help="outlier_distribution")
parser.add_argument("--pred-len", type=int, default=12)
parser.add_argument("--rho", type=float, default=0.01)
parser.add_argument("--diag", action="store_true")
parser.add_argument("--mse_coef", type=float, default=1)
parser.add_argument("--flow", action="store_true")
parser.add_argument('--nonlinearity', type=str, default='softplus', choices=["softmax", "softplus", "elu", "sigmoid", "exp"])
parser.add_argument('--loss', type=str, default='maskedmae', choices=["maskedmse", "maskedmae"])

args = parser.parse_args()

args.pred_len = [2, 5, 8, 11]
# args.pred_len = list(range(12))

os.chdir("/app")
result = pd.DataFrame(np.zeros((0, 4)))
sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
torch.manual_seed(99)
np.random.seed(99)

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
target_sensors = sensor_ids

num_nodes = len(target_sensors)

target_sensor_inds = [sensor_id_to_ind[i] for i in target_sensors]
flow = False
dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size,
                               target_sensor_inds=target_sensor_inds, flow=flow)

device = torch.device(args.device)
scaler = dataloader['scaler']
supports = [torch.tensor(i).to(device) for i in adj_mx]

if args.randomadj:
    adjinit = None
else:
    adjinit = supports[0]

if args.aptonly:
    supports = None


savedir = "/app/logspemsbay2022speed325/"
multistep = "GWNMDN_multistep_20220917-040847_N1_R325_reg0.0_nhid16_pred[2, 5, 8, 11]_rho0.0_diagTrue_msecoef1.0"
residual = "GWNMDN_residual_20220917-040849_N1_R10_reg0.0_nhid16_pred[2, 5, 8, 11]_rho0.001_diagFalse_msecoef1.0_nlinsoftplus"


engine_multistep = trainer_multistep(scaler, args.in_dim, args.seq_length, num_nodes, 5, 16, args.dropout,
                                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                                     adjinit, n_components=1, reg_coef=args.reg_coef, consider_neighbors=False,
                                     outlier_distribution=False, pred_len=[2, 5, 8, 11], rho=0, diag=True,
                                     mse_coef=1, loss=args.loss)

engine_residual = trainer_residual(scaler, args.in_dim, args.seq_length, num_nodes, 10, 16, args.dropout,
                                   args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                                   adjinit, n_components=1, reg_coef=0, pred_len=[2, 5, 8, 11], rho=0.001, diag=False,
                                   mse_coef=1)

engine_multistep.load(model_path=savedir + multistep + '/best_model.pt',
                      cov_path=savedir + multistep + '/best_covariance.pt',
                      fc_w_path=savedir + multistep + '/best_fc_w.pt')

engine_residual.load(model_path=savedir + residual + '/best_model_list.pt',)

result_multistep = []
result_residual = []
for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    # iter, (x, y) = next(enumerate(dataloader['test_loader'].get_iterator()))
    with torch.no_grad():
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        info_multistep = engine_multistep.eval(testx, testy[:, 0, :, :])
        info_residual = engine_residual.eval(testx, testy[:, 0, :, :])

        target = info_multistep["target"]
        mu_multistep = info_multistep["mu"]
        mu_residual = info_residual["mu"]

        target = info_multistep["scaler"].inverse_transform(target)
        mu_multistep = info_multistep["scaler"].inverse_transform(mu_multistep)
        mu_residual = info_residual["scaler"].inverse_transform(mu_residual)
        mu_multistep[mu_multistep < 0] = 0
        mu_residual[mu_residual < 0] = 0

        mu_multistep = mu_multistep.reshape(target.shape)
        mu_residual = mu_residual.reshape(target.shape)

        mask = (target > 0).float()

        mu_multistep = mu_multistep*mask + target*(1-mask)
        mu_residual = mu_residual*mask + target*(1-mask)
        result_multistep.append((target - mu_multistep)*mask)
        result_residual.append((target - mu_residual)*mask)


result_multistep = torch.concat(result_multistep, axis=0).cpu().numpy()
result_residual = torch.concat(result_residual, axis=0).cpu().numpy()

fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
for sensor in range(325):
    x1 = result_multistep[:, sensor, :].flatten()
    x2 = result_residual[:, sensor, :].flatten()
    n1, bins1, patches1 = plt.hist(x1, bins=1000, alpha=0.3, label="GWN")
    n2, bins2, patches2 = plt.hist(x2, bins=1000, alpha=0.3, label="GWN+STRR(10)")
    plt.grid(axis='y', alpha=0.75)
    plt.xlim([-20, 20])
    plt.legend(loc='upper right')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Sensor ' + str(sensor))

    fig.savefig(f"fig/hist/hist_{sensor}",  dpi='figure', format=None, metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    ax.cla()


fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
n, bins, patches = plt.hist(result_multistep.flatten(), bins=1000, alpha=0.3, label="GWN")
n, bins, patches = plt.hist(result_residual.flatten(), bins=1000, alpha=0.3, label="GWN+STRR(10)")
plt.grid(axis='y', alpha=0.75)
plt.xlim([-20, 20])
plt.legend(loc='upper right')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

plt.plot(result_multistep.flatten(), result_residual.flatten(), 'o', alpha=0.1)
plt.xlabel("GWN")
plt.ylabel("GWN+STRR(10)")
plt.show()


np.mean(np.abs(result_multistep[:, :, 3]))
np.mean(np.abs(result_residual[:, :, 3]))


fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

torch.where(target < 30)[1]

fig, ax = plt.subplots(nrows=2, ncols=2)  # create figure & 1 axis

for sensor in range(325):
    for i in range(4):
        ax[0, 0].plot(target[:, sensor, i].cpu().numpy(), color='black', label='target')
        ax[0, 0].plot(mu_multistep[:, sensor, i].cpu().numpy(), color='red', label='multistep')
        ax[0, 1].plot(target[:, sensor, i].cpu().numpy(), color='black', label='target')
        ax[0, 1].plot(mu_residual[:, sensor, i].cpu().numpy(), color='blue', label='residual')

        ax[1, 0].plot(target[:, sensor, i].cpu().numpy() - mu_multistep[:, sensor, i].cpu().numpy(), 'red')
        ax[1, 1].plot(target[:, sensor, i].cpu().numpy() - mu_residual[:, sensor, i].cpu().numpy(), color='blue')

        res_max = max((target[:, sensor, i].cpu().numpy() - mu_multistep[:, sensor, i].cpu().numpy()).max(),
                      (target[:, sensor, i].cpu().numpy() - mu_residual[:, sensor, i].cpu().numpy()).max())

        res_min = min((target[:, sensor, i].cpu().numpy() - mu_multistep[:, sensor, i].cpu().numpy()).min(),
                      (target[:, sensor, i].cpu().numpy() - mu_residual[:, sensor, i].cpu().numpy()).min())

        res_max = 1.1 * res_max if res_max > 0 else 0.9 * res_max
        res_min = 1.1 * res_min if res_min < 0 else 0.9 * res_min

        ax[0, 0].set_ylim(0, 80)
        ax[0, 1].set_ylim(0, 80)
        ax[1, 0].set_ylim(res_min, res_max)
        ax[1, 1].set_ylim(res_min, res_max)

        # ax.plot()
        # ax.legend(["target", "multistep", "residual"])
        # plt.ylim([0, 80])
        fig.savefig(f"fig/{sensor}_{i}",  dpi='figure', format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None)
        ax[0, 0].cla()
        ax[0, 1].cla()
        ax[1, 0].cla()
        ax[1, 1].cla()
