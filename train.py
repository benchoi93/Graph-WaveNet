import datetime
import os
import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/PEMS-BAY-delay3', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx_bay.pkl', help='adj data path')
# parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
# parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
# parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
# parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--num-rank', type=int, default=5, help='')
parser.add_argument('--nhid', type=int, default=128, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=325, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
# parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save', type=str, default='./garage/pems', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--n_components', type=int, default=5, help='')
parser.add_argument('--reg_coef', type=float, default=0.1, help='')
parser.add_argument('--save_every', type=int, default=20, help='')
parser.add_argument('--rho', type=float, default=0.0, help='')

parser.add_argument('--pred_seq', type=int, default=6, help='')
parser.add_argument('--delay', type=int, default=3, help='')

parser.add_argument('--train_L_batch', default=1, type=int, help='')
parser.add_argument('--train_L_space', default=1, type=int, help='')
parser.add_argument('--train_L_time', default=1, type=int, help='')

args = parser.parse_args()

args.gcn_bool = True
args.adjtype = 'doubletransition'
args.addaptadj = True
args.randomadj = True

args.train_L_batch = True if args.train_L_batch == 1 else False
args.train_L_space = True if args.train_L_space == 1 else False
args.train_L_time = True if args.train_L_time == 1 else False

args.batch_size = args.batch_size // args.delay

wandb.init(project="GWN_batch2", config=args,
           name=f"GWN_testbatch_space{args.train_L_space}_time{args.train_L_time}_batch{args.train_L_batch}_rho{args.rho}_delay{args.delay}")

save_dir = f"./model_save/GWN_testbatch_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_space{args.train_L_space}_time{args.train_L_time}_batch{args.train_L_batch}_rho{args.rho}_delay{args.delay}"

if os.path.exists(save_dir) is False:
    os.makedirs(save_dir, exist_ok=True)


def main():
    # set seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, delay=args.delay)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler=scaler,
                     in_dim=args.in_dim,
                     seq_length=args.seq_length,
                     num_nodes=args.num_nodes,
                     nhid=args.nhid,
                     dropout=args.dropout,
                     lrate=args.learning_rate,
                     wdecay=args.weight_decay,
                     device=device,
                     supports=supports,
                     gcn_bool=args.gcn_bool,
                     addaptadj=args.addaptadj,
                     aptinit=adjinit,
                     delay=args.delay,
                     train_L_space=args.train_L_space,
                     train_L_batch=args.train_L_batch,
                     train_L_time=args.train_L_time,
                     rho=args.rho,)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs+1):
        # if i % 10 == 0:
        # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # g['lr'] = lr
        # g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        train_mae = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = trainy.transpose(1, 3)

            train_data = torch.concat((trainx, trainy), dim=3)
            train_data = torch.stack([train_data[..., i:i+args.seq_length*2] for i in range(args.delay)], dim=1)
            trainx = train_data[..., :args.seq_length]
            trainy = train_data[..., args.seq_length:]

            metrics = engine.train(trainx, trainy[:, :, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1] * 100)
            train_rmse.append(metrics[2])
            train_mae.append(metrics[3])

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

            # break

        t2 = time.time()
        train_time.append(t2-t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        valid_mae = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = trainy.transpose(1, 3)

            train_data = torch.concat((trainx, trainy), dim=3)
            train_data = torch.stack([train_data[..., i:i+args.seq_length*2] for i in range(args.delay)], dim=1)
            trainx = train_data[..., :args.seq_length]
            trainy = train_data[..., args.seq_length:]

            metrics = engine.eval(trainx, trainy[:, :, 0, :, :])

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1] * 100)
            valid_rmse.append(metrics[2])
            valid_mae.append(metrics[3])
            # break

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_mae = np.mean(train_mae)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_mae = np.mean(valid_mae)
        his_loss.append(mvalid_loss)

        wandb.log(
            {
                "train/01_rmse": mtrain_rmse,
                "train/02_mape": mtrain_mape,
                "train/03_mae": mtrain_mae,
                "train/04_loss": mtrain_loss,
                "val/01_rmse": mvalid_rmse,
                "val/02_mape": mvalid_mape,
                "val/03_mae": mvalid_mae,
                "val/04_loss": mvalid_loss,
            },
            step=i,
        )

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)

        if i % args.save_every == 0:
            torch.save(engine.model.state_dict(), save_dir+f"/model_epoch_{i}.pth")
        # break
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    # engine.model.load_state_dict(torch.load(save_dir+f"/model_epoch_{bestid+1}.pth"))

    outputs = []
    realy = []
    # realy = torch.Tensor(dataloader['y_test']).to(device)
    # realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        testx = testx.transpose(1, 3)
        testy = testy.transpose(1, 3)

        test_data = torch.concat((testx, testy), dim=3)
        test_data = torch.stack([test_data[..., i:i+args.seq_length*2] for i in range(args.delay)], dim=1)
        testx = test_data[..., :args.seq_length]
        testy = test_data[..., args.seq_length:]

        with torch.no_grad():
            loss, preds = engine.process_batch(testx, testy[:, :, 0, :, :])
            # preds = engine.model(testx)
        outputs.append(preds.squeeze())
        realy.append(testy[:, :, 0, :, :].squeeze())
        # break

    yhat = torch.cat(outputs, dim=0)
    realy = torch.cat(realy, dim=0)

    yhat = yhat.reshape(yhat.shape[0] * yhat.shape[1], yhat.shape[2], yhat.shape[3])
    realy = realy.reshape(realy.shape[0] * realy.shape[1], realy.shape[2], realy.shape[3])

    print("Training finished")

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = yhat[:, :, i]
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1]*100)
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(), save_dir+"/model_best.pth")

    for i in range(12):
        wandb.log(
            {
                f"test/01_rmse_@{str(i).zfill(2)}": armse[i],
                f"test/02_mape_@{str(i).zfill(2)}": amape[i],
                f"test/03_mae_@{str(i).zfill(2)}": amae[i],
            }
        )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
