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
from tqdm import tqdm

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
parser.add_argument('--n_components', type=int, default=6, help='experiment id')
parser.add_argument('--reg_coef', type=float, default=0.1, help='experiment id')
parser.add_argument('--save_every', type=int, default=20, help='experiment id')
# parser.add_argument("--model_path", default="./logs_0614_predlen_test/GWN_MDN_20220614-163156_N6_R5_reg0.001_nhid4_pred3", type=str,  help="path to model")

args = parser.parse_args()

# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiFalse
# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiTrue

model_path = "logs/GWN_MDN_20220730-223937_N10_R5_reg0.0_nhid16_pred12_rho0.1"


def main(model_path, dataloader, adj_mx, target_sensors, target_sensor_inds, num_nodes):

    params = model_path.split("_")[3:]

    n_components = int(params[0].split("N")[1])
    num_rank = int(params[1].split("R")[1])
    reg_coef = float(params[2].split("reg")[1])
    nhid = int(params[3].split("nhid")[1])
    pred_len = int(params[4].split("pred")[1])
    rho = float(params[5].split("rho")[1])

    # set seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)

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

    engine = MDN_trainer(scaler, args.in_dim, args.seq_length, num_nodes, num_rank, nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, n_components=n_components, reg_coef=reg_coef, pred_len=pred_len,
                         rho=rho)

    engine.load(model_path=model_path + '/model.pt',
                cov_path=model_path + '/covariance.pt',
                fc_w_path=model_path + '/fc_w.pt')

    results = []
    for iter, (x, y) in tqdm(enumerate(dataloader['val_loader'].get_iterator()), total=dataloader['val_loader'].num_batch):
        with torch.no_grad():
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            info = engine.eval(testx, testy[:, 0, :, :])

        w = info["w"]
        w = w.exp()
        mus = info["mu"]
        L = info["scale_tril"]

        real_val = testy[:, 0, :, :]
        real = real_val[:, :, engine.pred_len - 1]
        target = testy[:, 0, :, :]

        dist = engine.mdn_head.get_output_distribution(features={'w': w, 'mu': mus, 'scale_tril': L, "target": target})

        output = engine.mdn_head.get_output_distribution(features={'w': w, 'mu': mus, 'scale_tril': L, "target": target}).mean
        predict = engine.scaler.inverse_transform(output)
        # (predict<0).sum()

        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        samples = engine.mdn_head.sample(features={'w': w, 'mu': mus, 'scale_tril': L, "target": target}, n=1000)
        pred_samples = engine.scaler.inverse_transform(samples)
        pred_samples[pred_samples < 0] = 0

        crps = torch.zeros(size=(y.shape[0], y.shape[2]))
        for i in range(pred_samples.shape[1]):
            for j in range(pred_samples.shape[2]):
                pred = pred_samples[:, i, j]
                crps[i, j] = ps.crps_ensemble(real[i, j].cpu().numpy(), pred)

        results.append(
            {
                "rmse": rmse,
                "mape": mape,
                "crps": crps,
                "mse": info['mse_loss'],
                "crps_mean": crps.mean().item(),
                "len": crps.shape[0],
                "nll": info["nll_loss"]
            }
        )

    # results = [{'rmse':x['rmse'] , 'mape':x['mape'] , 'crps' : x['crps'] , 'crps_mean':x['crps'].mean().item() , 'len' : x['crps'].shape[0]} for x in results]

    result_rmse = sum([x['rmse'] * x['len'] for x in results]) / sum([x['len'] for x in results])
    result_mape = sum([x['mape'] * x['len'] for x in results]) / sum([x['len'] for x in results]) * 100
    result_crps = sum([x['crps_mean'] * x['len'] for x in results]) / sum([x['len'] for x in results])
    result_nll = sum([x['nll'] * x['len'] for x in results]) / sum([x['len'] for x in results])

    return result_rmse, result_mape, result_crps, result_nll

    # for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    #     testx = torch.Tensor(x).to(device)
    #     testx = testx.transpose(1, 3)
    #     testy = torch.Tensor(y).to(device)
    #     testy = testy.transpose(1, 3)
    #     # break

    #     # metrics = engine.eval(input, testy[:, 0, :, :])
    #     with torch.no_grad():
    #         input = testx
    #         real_val = testy[:, 0, :, :]

    #         engine.model.train()
    #         engine.optimizer.zero_grad()
    #         input = nn.functional.pad(input, (1, 0, 0, 0))
    #         output = engine.model(input)
    #         output = output.transpose(1, 3)

    #         output = output.view(-1, engine.num_nodes, engine.n_components, engine.out_per_comp)
    #         L = torch.tril(engine.covariance.L.unsqueeze(0).expand(output.shape[0], -1, -1, -1))
    #         L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)] = torch.nn.functional.elu(
    #             L[:, :, torch.arange(engine.num_nodes), torch.arange(engine.num_nodes)]) + 1

    #         mus = output[:, :, :, 0]

    #         V = output[:, :, :, 1:]
    #         V = torch.einsum('abcd -> acbd', V)

    #         mus = torch.einsum('abc->acb', mus)
    #         output = output.reshape(-1, engine.n_components, engine.num_nodes * engine.out_per_comp)
    #         w = engine.fc_w(output)
    #         w = nn.functional.softmax(w, dim=1)

    #         scaled_real_val = engine.scaler.transform(real_val)
    #         loss, nll_loss, reg_loss = engine.mdn_head.forward(features={'w': w, 'mu': mus, 'scale_tril': L}, y=scaled_real_val)

    #     # (62): (62+288)
    #     # x.shape
    #     # real = real_val[:, :, 11]
    #     time_y = testy[(62):(62+288), 1, 0, -1]
    #     out_y = testy[(62):(62+288), 0, :, -1]

    #     mu_out = mus[(62): (62+288), :, :]
    #     w_out = w[(62): (62+288), :, 0]

    #     time_y.shape
    #     w_out.shape

    #     import pandas as pd
    #     wdata = pd.DataFrame(torch.concat((time_y.unsqueeze(1), w_out), dim=1).numpy())
    #     columns = ['time']
    #     for i in range(6):
    #         columns += [f"w{i}"]
    #     wdata.columns = columns

    #     sns.set()
    #     fig, axes = plt.subplots(1, 6, figsize=(30, 5))

    #     for i in range(6):
    #         sns.lineplot(data=wdata, x='time', y=f'w{i}', ax=axes[i])

    #     plt.savefig(f"{args.model_path}/w_out.png")

    #     # sns.relplot(x="time", y=['w1', 'w2', 'w3', 'w4', 'w5'], data=wdata)

    #     features = {'w': w, 'mu': mus, 'scale_tril': L}

    #     dist = engine.mdn_head.get_output_distribution(features)
    #     sample_cov = dist.component_distribution.covariance_matrix[0]
    #     sample_prec = dist.component_distribution.precision_matrix[0]

    #     corr = torch.zeros_like(sample_cov)
    #     for i in range(sample_cov.size(0)):
    #         corr[i] = torch.corrcoef(sample_cov[i])

    #     sparsity = (sample_prec.abs() > 0.01).float()


    #     fig, axes = plt.subplots(1 , 6, figsize=(35, 5))
    #     for i in range(6):
    #         ax = axes[i]
    #         m = ax.pcolormesh(
    #             sample_cov[i].detach().cpu().numpy(), cmap='coolwarm'
    #         )
    #     fig.colorbar(m, label="covariance")
    #     plt.savefig(f"{args.model_path}/cov_out.png")
    #     break
if __name__ == "__main__":
    import os
    import pandas as pd
    result = pd.DataFrame(np.zeros((0, 4)))
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

    num_nodes = len(target_sensors)

    target_sensor_inds = [sensor_id_to_ind[i] for i in target_sensors]

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, target_sensor_inds=target_sensor_inds)

    # file_list = [x for x in os.listdir("logs")]
    file_list = [x for x in os.listdir("logs") if "0808" in x]
    # file_list = [x for x in os.listdir("logs") if "GWN_MDNdiag_20220623" in x]
    # file_list = ["GWN_MDNdiag_20220624-132309_N1_R5_reg0.01_nhid4_pred36", "GWN_MDNdiag_20220624-132314_N1_R5_reg0.001_nhid4_pred36"]
    print(file_list)
    for file in file_list:
        print(file)
        # try:
        result_rmse, result_mape, result_crps, result_nll = main(
            f"logs/{file}", dataloader, adj_mx, target_sensors, target_sensor_inds, num_nodes)

        result = pd.concat([result, pd.DataFrame([[file, result_rmse, result_mape, result_crps, result_nll]])], axis=0)
        result.to_csv("out_with_nll_0808.csv")
        # except:
        #     print(file)
        #     continue
