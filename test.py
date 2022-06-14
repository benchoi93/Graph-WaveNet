import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
# from mdn_engine import MDN_trainer
from Fixed_mdn_engine import MDN_trainer
import torch.nn as nn
import seaborn as sns

import sys
sys.argv = ['']
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='')
parser.add_argument('--data', type=str, default='data/PEMS-BAY', help='data path')
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
parser.add_argument('--n_components', type=int, default=5, help='experiment id')
parser.add_argument('--reg_coef', type=float, default=0.1, help='experiment id')
parser.add_argument('--save_every', type=int, default=20, help='experiment id')
parser.add_argument("--model_path", type=str, default='./logs/GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiFalse', help="path to model")

args = parser.parse_args()

# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiFalse
# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiTrue


def main():
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
        # break

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
            L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)] = torch.nn.functional.elu(
                L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)]) + 1

            mus = output[:, :, :, 0]

            V = output[:, :, :, 1:]
            V = torch.einsum('abcd -> acbd', V)

            mus = torch.einsum('abc->acb', mus)
            output = output.reshape(-1, engine.n_components, engine.num_nodes * engine.out_per_comp)
            w = engine.fc_w(output)
            w = nn.functional.softmax(w, dim=1)

            scaled_real_val = engine.scaler.transform(real_val)
            loss, nll_loss, reg_loss = engine.mdn_head.forward(features={'w': w, 'mu': mus, 'scale_tril': L}, y=scaled_real_val)

        # (62): (62+288)
        # x.shape
        # real = real_val[:, :, 11]
        time_y = testy[(62):(62+288), 1, 0, -1]
        out_y = testy[(62):(62+288), 0, :, -1]

        mu_out = mus[(62): (62+288), :, :]
        w_out = w[(62): (62+288), :, 0]

        time_y.shape
        w_out.shape

        import pandas as pd
        wdata = pd.DataFrame(torch.concat((time_y.unsqueeze(1), w_out), dim=1).numpy())
        wdata.columns = ['time', 'w1', 'w2', 'w3', 'w4', 'w5']

        sns.set()
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        sns.lineplot(data=wdata, x='time', y='w1', ax=axes[0])
        sns.lineplot(data=wdata, x='time', y='w2', ax=axes[1])
        sns.lineplot(data=wdata, x='time', y='w3', ax=axes[2])
        sns.lineplot(data=wdata, x='time', y='w4', ax=axes[3])
        sns.lineplot(data=wdata, x='time', y='w5', ax=axes[4])

        plt.savefig(f"w_out.png")

        # sns.relplot(x="time", y=['w1', 'w2', 'w3', 'w4', 'w5'], data=wdata)

        features = {'w': w, 'mu': mus, 'scale_tril': L}

        dist = engine.mdn_head.get_output_distribution(features)
        sample_cov = dist.component_distribution.covariance_matrix[0]
        sample_prec = dist.component_distribution.precision_matrix[0]

        corr = torch.zeros_like(sample_cov)
        for i in range(sample_cov.size(0)):
            corr[i] = torch.corrcoef(sample_cov[i])

        sparsity = (sample_prec.abs() > 0.01).float()

        fig, axes = plt.subplots(1, 5, figsize=(30, 5))
        for i in range(len(axes)):
            ax = axes[i]
            m = ax.pcolormesh(
                sample_cov[i].detach().cpu().numpy(), cmap='coolwarm'
            )
        # sns_plot = sns.heatmap(sample_cov[0].detach().cpu().numpy(), cmap='coolwarm', ax=axes[0])
        # sns_plot = sns.heatmap(sample_cov[1].detach().cpu().numpy(), cmap='coolwarm', ax=axes[1])
        # sns_plot = sns.heatmap(sample_cov[2].detach().cpu().numpy(), cmap='coolwarm', ax=axes[2])
        # sns_plot = sns.heatmap(sample_cov[3].detach().cpu().numpy(), cmap='coolwarm', ax=axes[3])
        # sns_plot = sns.heatmap(sample_cov[4].detach().cpu().numpy(), cmap='coolwarm', ax=axes[4])
        fig.colorbar(m, label="covariance")
        plt.savefig(f"cov_out.png")
        break


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
