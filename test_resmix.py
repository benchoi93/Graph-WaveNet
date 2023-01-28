import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
# from mdn_engine import MDN_trainer
# from Fixed_mdn_engine import MDN_trainer
from Fixed_mdn_engine_resmix import MDN_trainer
# from Diag_Fixed_mdn_engine import MDN_trainer
import torch.nn as nn
import seaborn as sns
import properscoring as ps
from tqdm import tqdm

import sys
sys.argv = ['']
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--data', type=str, default='data/PEMS-BAY', help='data path')
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
parser.add_argument('--nhid', type=int, default=4, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100000, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./garage/pems', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--n_components', type=int, default=6, help='experiment id')
parser.add_argument('--reg_coef', type=float, default=0.1, help='experiment id')
parser.add_argument('--save_every', type=int, default=20, help='experiment id')
# parser.add_argument("--model_path", default="./logs_0614_predlen_test/GWN_MDN_20220614-163156_N6_R5_reg0.001_nhid4_pred3", type=str,  help="path to model")
# parser.add_argument("--pred-len", type=int, default=12)
parser.add_argument("--rho", type=float, default=0.1)
parser.add_argument("--diag", action="store_true")
parser.add_argument("--mse_coef", type=float, default=1)
parser.add_argument("--flow", action="store_true")

args = parser.parse_args()

# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiFalse
# GWN_MDN_20220607-133432_N5_R5_reg0.001_nhid4_neiTrue

# model_path = "logsisttt1121pemsbay2017speed/GWNMDN_ResMix_20221120-221426maeloss_N1_R1_reg0.0_nhid64_pred12_rho0.0_diagFalse_msecoef1.0_nlinsoftplus"
model_path = "logsisttt1121pemsbay2017speed/GWNMDN_ResMix_20221120-221425maeloss_N1_R3_reg0.0_nhid64_pred12_rho0.001_diagFalse_msecoef1.0_nlinsoftplus"

# --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
args.gcn_bool = True
args.addaptadj = True
args.adjtype = "doubletransition"
args.randomadj = True


def main(model_path, dataloader, adj_mx, target_sensors, target_sensor_inds, num_nodes):

    params = model_path.split("_")[3:]

    if "mse" in model_path.split("_")[2]:
        loss = "mse"
    elif "mae" in model_path.split("_")[2]:
        loss = "mae"
    else:
        raise ValueError("loss type not found")

    n_components = int(params[0].split("N")[1])
    num_rank = int(params[1].split("R")[1])
    reg_coef = float(params[2].split("reg")[1])
    nhid = int(params[3].split("nhid")[1])
    pred_len = int(params[4].split("pred")[1])
    if pred_len == 4:
        pred_len = [2, 5, 8, 11]
    elif pred_len == 12:
        pred_len = list(range(12))
    rho = float(params[5].split("rho")[1])

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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

    # adjinit = adjinit[:, target_sensor_inds][target_sensor_inds, :]

    # engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
    #                  args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
    #                  adjinit)

    engine = MDN_trainer(scaler, args.in_dim, args.seq_length, num_nodes, num_rank, nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, n_components=n_components, reg_coef=reg_coef, pred_len=pred_len, rho=rho, diag=args.diag,
                         mse_coef=args.mse_coef, summary=False, loss=loss)

    # engine = MDN_trainer(scaler, args.in_dim, args.seq_length, num_nodes, num_rank, args.nhid, args.dropout,
    #                      args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
    #                      adjinit, n_components=n_components, reg_coef=reg_coef, consider_neighbors=args.consider_neighbors,
    #                      outlier_distribution=args.outlier_distribution, pred_len=pred_len, rho=rho, diag=args.diag,
    #                      mse_coef=args.mse_coef, nonlinearity=args.nonlinearity, loss=loss)

    engine.load(model_path=model_path + '/best_model_list.pt')

    engine.model.eval()
    # engine.covariance.eval()
    # engine.fc_ws.eval()
    # engine.fc_wt.eval()
    L_s, L_t = engine.get_L()

    prc_s = torch.einsum("bij,bjk->bik", L_s, L_s.transpose(-1, -2))
    prc_t = torch.einsum("bij,bjk->bik", L_t, L_t.transpose(-1, -2))

    # dump pickle
    import pickle
    with open(model_path + "/prc_s.pkl", "wb") as f:
        pickle.dump(prc_s, f)
    with open(model_path + "/prc_t.pkl", "wb") as f:
        pickle.dump(prc_t, f)

    cov_s = torch.inverse(prc_s)
    cov_t = torch.inverse(prc_t)

    cor_s = torch.zeros_like(cov_s)
    cor_t = torch.zeros_like(cov_t)

    for i in range(cov_s.shape[0]):
        cor_s[i] = cov_s[i] / (torch.sqrt(torch.diag(cov_s[i]))[:, None] * torch.sqrt(torch.diag(cov_s[i]))[None, :])
        cor_t[i] = cov_t[i] / (torch.sqrt(torch.diag(cov_t[i]))[:, None] * torch.sqrt(torch.diag(cov_t[i]))[None, :])

    sns.heatmap(cov_t[0].detach().cpu().numpy(), vmin=-cov_t.min(), vmax=cov_t.max(), cmap="RdBu_r")
    plt.show()
    sns.heatmap(cov_t[1].detach().cpu().numpy(), vmin=-cov_t.min(), vmax=cov_t.max(), cmap="RdBu_r")
    plt.show()
    sns.heatmap(cov_t[2].detach().cpu().numpy(), vmin=-cov_t.min(), vmax=cov_t.max(), cmap="RdBu_r")
    plt.show()

    sns.heatmap(cov_s[0].detach().cpu().numpy(), vmin=-cov_s.min(), vmax=cov_s.max(), cmap="RdBu_r")
    plt.show()
    sns.heatmap(cov_s[1].detach().cpu().numpy(), vmin=-cov_s.min(), vmax=cov_s.max(), cmap="RdBu_r")
    plt.show()
    sns.heatmap(cov_s[2].detach().cpu().numpy(), vmin=-cov_s.min(), vmax=cov_s.max(), cmap="RdBu_r")
    plt.show()

    sns.heatmap(cor_t[0].detach().cpu().numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    plt.show()
    sns.heatmap(cor_t[1].detach().cpu().numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    plt.show()
    sns.heatmap(cor_t[2].detach().cpu().numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    plt.show()

    results = []
    w_list = []
    test_x_list = []
    test_y_list = []
    mu_list = []
    target_list = []
    # crps_list = []
    for iter, (x, y) in tqdm(enumerate(dataloader['test_loader'].get_iterator()), total=dataloader['test_loader'].num_batch):
        with torch.no_grad():
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            # info = engine.train(testx, testy[:, 0, :, :], eval=True)
            info = engine.eval(testx, testy[:, 0, :, :])
            # crps_list.append(info["crps"])

        w_list.append(info["w"].detach().cpu())
        test_x_list.append(testx.detach().cpu())
        test_y_list.append(testy.detach().cpu())
        # testx[74:74+288, :, 0, 0]

        # plt.plot(info['w'][74:74+288][:, 0, :].cpu().numpy(), label='w0')
        # plt.plot(info['w'][74:74+288][:, 1, :].cpu().numpy(), label='w1')
        # plt.plot(info['w'][74:74+288][:, 2, :].cpu().numpy(), label='w2')
        # plt.legend()
        # plt.show()

        mu = info['mu'].reshape(info['mu'].shape[0], num_nodes, engine.num_pred)
        mu = engine.scaler.inverse_transform(mu)
        mu[mu < 0] = 0
        target = testy[:, 0, :, pred_len]
        info["target"] = engine.scaler.transform(target).reshape(target.shape[0], -1)
        mu_list.append(mu.detach().cpu())
        target_list.append(target.detach().cpu())

        # crps, ES = engine.get_crps(info)
        # crps = crps.unsqueeze(1).expand((crps.shape[0], num_nodes, engine.num_pred))
        # crps = crps.reshape(crps.shape[0], num_nodes, engine.num_pred)
        # crps = torch.FloatTensor(crps).to(device)

        results_timestep = []
        for i in range(engine.num_pred):
            mask = target[:, :, i] > 0

            rmse = ((((mu[:, :, i]-target[:, :, i])**2) * mask).sum() / mask.sum()).sqrt()
            mape = ((((mu[:, :, i] - target[:, :, i]).abs() / (target[:, :, i] + 1e-6)) * mask).sum() / mask.sum()) * 100
            mae = ((mu[:, :, i] - target[:, :, i]).abs() * mask).sum() / mask.sum()

            # crps_i = (crps[:, :, i] * mask).sum() / mask.sum()
            # print(f"{mask.sum() / (mask.shape[0]*mask.shape[1])*100:.2f}")
            # ES_i = ES[(target == 0).sum((-1, -2)) == 0].mean()

            # results_timestep.append([rmse.item(), mape.item(), mae.item(), crps_i.item(), ES_i.item()])
            results_timestep.append([rmse.item(), mape.item(), mae.item()])

        results.append(
            [results_timestep, x.shape[0]]
        )

    w_list = torch.concat(w_list, 0)

    w_list.squeeze_()
    test_x[:, 1, 0, -1].shape

    out = torch.concat([w_list, test_x[:, 1, 0, -1].unsqueeze(-1)], 1)
    torch.save(out, 'w_list.pt')

    # test_x_list = np.concatenate(test_x_list, 0)

    # tod = test_x_list[:, 1, 0, 0]

    # tod[4970:4970+288]

    # plt.plot(w_list[4970:4970+288, 0, :], label='w0')
    # plt.plot(w_list[4970:4970+288, 1, :], label='w1')
    # plt.plot(w_list[4970:4970+288, 2, :], label='w2')
    # plt.legend()
    # plt.show()

    test_y = torch.concat(test_y_list, 0)
    test_x = torch.concat(test_x_list, 0)
    mu = torch.concat(mu_list, 0)
    target = torch.concat(target_list, 0)

    test_y[0, 1, 0, 0]

    test_y = test_y[62:]

    test_y = test_y[62:10142]
    mu = mu[62:10142]
    target = target[62:10142]

    target = target.reshape(35, 288, 325, 12)
    mu = mu.reshape(35, 288, 325, 12)
    mu[target[:, :, :, :] == 0] = 0

    # res = (info['scaler'].transform(test_y[:, :, 0, :, :]) - info['scaler'].transform(mu))
    res = info['scaler'].transform(target) - info['scaler'].transform(mu)

    (res[:, t*12, i, :].T @ res[:, t*12, i, :])

    cov_total = torch.zeros(325, 24, 12, 12)
    for i in tqdm(range(325)):
        # i=7
        for t in range(23):
            # plot covariance = res[:,t,i,:].T @ res[:,t,i,:]
            # save it into /app/spatial_cov/loc_{i}_time_{t}.png
            restemp = res[:, t*12:(t+1)*12, i, :]
            restemp = restemp.reshape(35*12, 12)
            covariance = restemp.T @ restemp
            cov_total[i, t] = covariance

    # cov_total.min()
    # cov_total.max()

    # i=7
    for i in tqdm(range(325)):
        for t in range(24):
            plt.imshow(cov_total[i, t], vmin=-cov_total[i].max(), vmax=cov_total[i].max(), cmap='RdBu_r')
            # add title
            plt.title(f'loc_{i}_time_{str(t).zfill(2)}:00')
            # add legend
            plt.colorbar()
            plt.savefig(f'/app/spatial_cov/loc_{i}_time_{str(t).zfill(2)}.png')
            plt.clf()

            correlation = torch.corrcoef(cov_total[i, t])
            plt.imshow(correlation, vmin=-1, vmax=1, cmap='RdBu_r')
            # add title
            plt.title(f'loc_{i}_time_{str(t).zfill(2)}:00')
            # add legend
            plt.colorbar()
            plt.savefig(f'/app/spatial_cor/loc_{i}_time_{str(t).zfill(2)}.png')
            plt.clf()

    cov_total = torch.zeros(12, 24, 325, 325)
    for i in tqdm(range(12)):
        for t in range(24):
            restemp = res[:, t*12:(t+1)*12, :, i]
            restemp = restemp.reshape(35*12, 325)
            covariance = restemp.T @ restemp
            cov_total[i, t] = covariance

    for i in tqdm(range(12)):
        for t in range(24):
            plt.imshow(cov_total[i, t], vmin=-cov_total[i].max(), vmax=cov_total[i].max(), cmap='RdBu_r')
            # add title
            plt.title(f'loc_{i}_time_{str(t).zfill(2)}:00')
            # add legend
            plt.colorbar()
            plt.savefig(f'/app/temporal_cov/loc_{i}_time_{str(t).zfill(2)}.png')
            plt.clf()

            correlation = torch.corrcoef(cov_total[i, t])
            plt.imshow(correlation, vmin=-1, vmax=1, cmap='RdBu_r')
            # add title
            plt.title(f'loc_{i}_time_{str(t).zfill(2)}:00')
            # add legend
            plt.colorbar()
            plt.savefig(f'/app/temporal_cor/loc_{i}_time_{str(t).zfill(2)}.png')
            plt.clf()

    # results = [{'rmse':x['rmse'] , 'mape':x['mape'] , 'crps' : x['crps'] , 'crps_mean':x['crps'].mean().item() , 'len' : x['crps'].shape[0]} for x in results]

    results_item = np.stack([np.stack(x[0]) for x in results])
    result_rmse = np.nanmean(results_item[:, :, 0], (0))
    result_mape = np.nanmean(results_item[:, :, 1], (0))
    result_mae = np.nanmean(results_item[:, :, 2], (0))
    # result_crps = np.nanmean(results_item[:, :, 3], (0))
    # result_ES = np.nanmean(results_item[:, :, 4], (0))

    # return result_rmse, result_mape, result_mae, result_crps, result_ES
    return result_rmse, result_mape, result_mae


fig = sns.heatmap(cor, cmap="bwr", vmin=-1, vmax=1)
plt.savefig('correlation.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()


fig = sns.heatmap(np.diag(np.ones(3899), 1) +
                  np.diag(np.ones(3899), -1) +
                  np.diag(np.ones(3898), 2) +
                  np.diag(np.ones(3898), -2) +
                  np.diag(np.ones(3900)),
                  cmap="bwr", vmin=-1, vmax=1)
plt.savefig('correlation_diag.png', bbox_inches='tight', dpi=1000)
plt.clf()
plt.cla()


if __name__ == "__main__":
    import os
    os.chdir("/app")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    import pandas as pd
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
    logpath = "logs"
    savepath = f"out_{logpath}.csv"

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size,
                                   target_sensor_inds=target_sensor_inds, flow=flow)

    file_list = [x for x in os.listdir(logpath) if "ResMix" in x]
    # file_list = [x for x in os.listdir("logs") if "0808" in x]
    # file_list = [x for x in os.listdir("logs") if "GWN_MDNdiag_20220623" in x]
    # file_list = ["GWN_MDNdiag_20220624-132309_N1_R5_reg0.01_nhid4_pred36", "GWN_MDNdiag_20220624-132314_N1_R5_reg0.001_nhid4_pred36"]
    print(file_list)
    for file in tqdm(file_list):
        print(file)
        # try:
        result_rmse, result_mape, result_mae = main(
            f"{logpath}/{file}", dataloader, adj_mx, target_sensors, target_sensor_inds, num_nodes)

        result = pd.concat([result, pd.DataFrame([[file, *result_rmse, *result_mape, *result_mae]])], axis=0)
        result.to_csv(savepath)
