import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
# from mdn_engine import MDN_trainer
# from Fixed_mdn_engine import MDN_trainer
from Fixed_mdn_engine_tensor import MDN_trainer
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
#parser.add_argument('--seed',type=int,default=99,help='random seed')
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

# model_path = "logs/GWNMDN_residual_20220825-094251_N8_R12_reg0.0_nhid16_pred[2, 5, 8, 11]_rho0.01_diagFalse_msecoef1.0"

# --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
args.gcn_bool = True
args.addaptadj = True
args.adjtype = "doubletransition"
args.randomadj = True


def main(model_path, dataloader, adj_mx, target_sensors, target_sensor_inds, num_nodes):

    params = model_path.split("_")[3:]

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

    # adjinit = adjinit[:, target_sensor_inds][target_sensor_inds, :]

    # engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
    #                  args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
    #                  adjinit)

    engine = MDN_trainer(scaler, args.in_dim, args.seq_length, num_nodes, num_rank, nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, n_components=n_components, reg_coef=reg_coef, pred_len=pred_len, rho=rho, diag=args.diag,
                         mse_coef=args.mse_coef, summary=False)

    engine.load(model_path=model_path + '/best_model_list.pt')

    engine.model.eval()
    # engine.covariance.eval()
    # engine.fc_ws.eval()
    # engine.fc_wt.eval()

    results = []
    for iter, (x, y) in tqdm(enumerate(dataloader['test_loader'].get_iterator()), total=dataloader['test_loader'].num_batch):
        with torch.no_grad():
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            # info = engine.train(testx, testy[:, 0, :, :], eval=True)
            info = engine.eval(testx, testy[:, 0, :, :])

        pred = info['pred'].reshape(info['pred'].shape[0], num_nodes, engine.num_pred)
        pred[pred < 0] = 0

        # mu = info['mu'].reshape(info['mu'].shape[0], num_nodes, engine.num_pred)
        # mu = scaler.inverse_transform(mu)
        # mu[mu < 0] = 0
        # pred = mu

        target = testy[:, 0, :, pred_len]
        info["target"] = engine.scaler.transform(target).reshape(target.shape[0], -1)

        results_timestep = []
        for i in range(engine.num_pred):
            mask = target[:, :, i] > 0

            rmse = ((((pred[:, :, i]-target[:, :, i])**2) * mask).sum() / mask.sum()).sqrt()
            mape = ((((pred[:, :, i] - target[:, :, i]).abs() / (target[:, :, i] + 1e-6)) * mask).sum() / mask.sum()) * 100
            mae = ((pred[:, :, i] - target[:, :, i]).abs() * mask).sum() / mask.sum()

            results_timestep.append([rmse.item(), mape.item(), mae.item()])

        results.append(
            [results_timestep, x.shape[0]]
        )

    # results = [{'rmse':x['rmse'] , 'mape':x['mape'] , 'crps' : x['crps'] , 'crps_mean':x['crps'].mean().item() , 'len' : x['crps'].shape[0]} for x in results]

    results_item = np.stack([np.stack(x[0]) for x in results])
    result_rmse = np.nanmean(results_item[:, :, 0], (0))
    result_mape = np.nanmean(results_item[:, :, 1], (0))
    result_mae = np.nanmean(results_item[:, :, 2], (0))

    return result_rmse, result_mape, result_mae


if __name__ == "__main__":
    import os
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

    file_list = [x for x in os.listdir(logpath) if "tensor" in x]
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
