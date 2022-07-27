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
parser.add_argument('--data', type=str, default='data/PEMS-BAY-3hr', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx_bay.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--num-rank', type=int, default=5, help='')
parser.add_argument('--nhid', type=int, default=4, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100000, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save', type=str, default='./garage/pems', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--n_components', type=int, default=10, help='experiment id')
parser.add_argument('--reg_coef', type=float, default=0.1, help='experiment id')
parser.add_argument('--save_every', type=int, default=20, help='experiment id')
parser.add_argument("--model_path", type=str, default='./logs/GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiFalse', help="path to model")

args = parser.parse_args()

# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiFalse
# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiTrue

# GWNMDN = "./logs_TRB_12sensors/GWN_MDN_20220620-200048_N10_R5_reg0.0_nhid4_pred12"
# GWNMDNreg = "./logs_TRB_12sensors/GWN_MDN_20220629-180633_N10_R5_reg0.1_nhid4_pred12"
# GWNMDNdiag = "./logs_TRB_12sensors/GWN_MDNdiag_20220619-145112_N10_R5_reg0.0_nhid4_pred12"
flist = os.listdir("logs")

# GWNMDN = flist[0]
GWNMDN = 'logs_TRB_12sensors/GWN_MDNdiag_20220726-170914_N10_R5_reg0.1_nhid16_pred12'
args.n_components = 10
args.nhid = 16
args.model_path = GWNMDN

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
                     adjinit, n_components=args.n_components, reg_coef=args.reg_coef)

engine.load(model_path=args.model_path + '/model.pt',
            cov_path=args.model_path + '/covariance.pt',
            fc_w_path=args.model_path + '/fc_w.pt')

for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    testx = testx.transpose(1, 3)
    testy = torch.Tensor(y).to(device)
    testy = testy.transpose(1, 3)

    # metrics = engine.eval(input, testy[:, 0, :, :])
    with torch.no_grad():
        input = testx
        real_val = testy[:, 0, :, :]

        engine.model.train()
        engine.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = engine.model(input)
        output = output.transpose(1, 3)

        output = output.view(-1, engine.num_nodes, engine.n_components, engine.out_per_comp)
        L = torch.tril(engine.covariance.L.unsqueeze(0).expand(output.shape[0], -1, -1, -1))
        # L = (torch.diag_embed(torch.ones(12)) * np.log(0.01)).expand(output.shape[0], 10, -1, -1)
        L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)] = torch.nn.functional.elu(
            L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)]) + 1

        mus = output[:, :, :, 0]
        mus = torch.stack([mus[:, :, 0]] * mus.shape[-1], -1)

        V = output[:, :, :, 1:]
        V = torch.einsum('abcd -> acbd', V)

        mus = torch.einsum('abc->acb', mus)
        output = output.reshape(-1, engine.n_components, engine.num_nodes * engine.out_per_comp)
        w = engine.fc_w(output)
        w = nn.functional.softmax(w, dim=1)

        scaled_real_val = engine.scaler.transform(real_val)
        loss, nll_loss, reg_loss, mse_loss = engine.mdn_head.forward(features={'w': w, 'mu': mus, 'scale_tril': L}, y=scaled_real_val)

        dist = engine.mdn_head.get_output_distribution(features={'w': w, 'mu': mus, 'scale_tril': L})
        nll_loss = - dist.log_prob(scaled_real_val[:, :, 11])
        nll_loss.shape

    # (62): (62+288)
    # x.shape
    # real = real_val[:, :, 11]
    time_y = testy[(46):(46+288), 1, 0, -1]
    out_y = testy[(46):(46+288), 0, :, -1]

    mu_out = mus[(46): (46+288), :, :]
    w_out = w[(46): (46+288), :, 0]
    L = torch.tril(engine.covariance.L.unsqueeze(0))
    L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)] = torch.nn.functional.elu(
        L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)]) + 1

    result = []
    for i in tqdm(range(time_y.shape[0])):
        time_y_i = time_y[i:(i+1)]
        out_y_i = out_y[i:(i+1)]
        mu_out_i = mu_out[i:(i+1), :]
        w_out_i = w_out[i:(i+1), :]

        engine.mdn_head.outlier_distribution = False
        dist_i = engine.mdn_head.get_output_distribution(features={
            'w': w_out_i,
            'mu': mu_out_i,
            'scale_tril': L
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
            mu_out_i_scaled = engine.scaler.inverse_transform(mu_out_i[0, 0, c].cpu().numpy())
            result.append([i, time_y_i.item(), c, out_y_i[0, c].item(), np.log(score), mean_y_i, mu_out_i_scaled])

    import pandas as pd
    result_pd = pd.DataFrame(result, columns=['time', 'time_y', 'c', 'out_y', 'score', 'sample_mean', 'mu'])
    result_pd.to_csv('crps_result_MDNdiag.csv', index=False)
    # result_pd.to_csv('crps_result_MDN_reg0.csv', index=False)
    break


result_pd_ind = pd.read_csv("crps_result_MDNind.csv")
result_pd_diag = pd.read_csv("crps_result_MDNdiag.csv")
result_pd_fullcov = pd.read_csv("crps_result_MDNfullcov_reg0.csv")


plt.scatter(result_pd_ind["score"], result_pd_fullcov["score"], alpha=0.1)
# draw x=y line
plt.plot([0, 5], [0, 5], 'k--')
plt.xlabel("MDN ind cov")
plt.ylabel("MDN full cov")
plt.xlim([-1, 5])
plt.ylim([-1, 5])
plt.show()

result_pd_ind['score'] = np.exp(result_pd_ind['score'])
result_pd_diag['score'] = np.exp(result_pd_diag['score'])
result_pd_fullcov['score'] = np.exp(result_pd_fullcov['score'])


# sns.set_theme(style="darkgrid")
sns.heatmap(
    result_pd_ind.pivot(index="time", columns="c", values="score"), cmap="BuGn"
)
plt.show()

# sns.set_theme(style="darkgrid")
sns.heatmap(
    result_pd_fullcov.pivot(index="time", columns="c", values="score"), cmap="BuGn"
)
plt.show()


# sns.set_theme(style="darkgrid")
sns.heatmap(
    result_pd_diag.pivot(index="time", columns="c", values="score"), cmap="BuGn", vmin=-3, vmax=5
)
plt.show()
# sns.relplot(x="out_y", y="score", hue="c",
#             sizes=(40, 400), alpha=.5, palette="muted",
#             height=6, data=result_pd)
# plt.show()


# sns.relplot(x="out_y", y="sample_mean", hue="c",
#             sizes=(40, 400), alpha=.5, palette="muted",
#             height=6, data=result_pd)
# plt.show()
# result_pd.score.mean()
