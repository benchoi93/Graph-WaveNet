import torch
import numpy as np
import argparse
import time
import util
# import matplotlib.pyplot as plt
# from engine import trainer
from engine_new import MDN_trainer
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')

parser.add_argument('--data', type=str, default='data/PEMS-BAY', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx_bay.pkl', help='adj data path')

parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=64, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=2, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./garage/pems', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--n_components', type=int, default=3, help='experiment id')
parser.add_argument('--save_every', type=int, default=20, help='experiment id')
parser.add_argument("--pred-len", type=int, default=12)
parser.add_argument("--rho", type=float, default=1)
parser.add_argument("--diag", action="store_true")
parser.add_argument('--loss', type=str, default='mse', choices=["mse", "mae"])
parser.add_argument('--mix_mean', type=str, default="False", choices=["True", "False"])

args = parser.parse_args()
args.mix_mean = True if args.mix_mean == "True" else False
wandb.init(project="GWN_0421", config=args)

print(wandb.config)

args.pred_len = list(range(12))

torch.backends.cudnn.benchmark = True


def main():
    args.n_components = wandb.config.n_components
    args.mix_mean = wandb.config.mix_mean
    args.rho = wandb.config.rho
    args.seed = wandb.config.seed

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)

    args.num_nodes = len(sensor_ids)

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    # engine = MDN_trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.n_components, args.nhid, args.dropout,
    #                      args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
    #                      adjinit, pred_len=args.pred_len, rho=args.rho, diag=args.diag, loss=args.loss)

    engine = MDN_trainer(scaler, args, device, supports, adjinit)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    best_val_loss = float('inf')
    for i in range(0, args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        train_mse_loss = []
        train_nll_loss = []
        t1 = time.time()
        dataloader['train_loader']
        for iter, (x, y) in enumerate(dataloader['train_loader']):
            trainx = x.to(device)
            trainx = trainx.transpose(1, 3)
            trainy = y.to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics['loss'])
            train_mape.append(metrics['mape'])
            train_rmse.append(metrics['rmse'])
            train_nll_loss.append(metrics['nll_loss'])
            train_mse_loss.append(metrics['mse_loss'])

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        valid_nll_loss = []
        valid_mse_loss = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader']):
            testx = x.to(device)
            testx = testx.transpose(1, 3)
            testy = y.to(device)
            testy = testy.transpose(1, 3)

            metrics = engine.eval(testx, testy[:, 0, :, :])

            valid_loss.append(metrics['loss'])
            valid_mape.append(metrics['mape'])
            valid_rmse.append(metrics['rmse'])
            valid_nll_loss.append(metrics['nll_loss'])
            valid_mse_loss.append(metrics['mse_loss'])

        test_loss = []
        test_mape = []
        test_rmse = []
        test_nll_loss = []
        test_mse_loss = []

        test_mape_list = []
        test_rmse_list = []
        test_mae_list = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['test_loader']):
            testx = x.to(device)
            testx = testx.transpose(1, 3)
            testy = y.to(device)
            testy = testy.transpose(1, 3)

            metrics = engine.eval(testx, testy[:, 0, :, :])

            test_loss.append(metrics['loss'])
            test_mape.append(metrics['mape'])
            test_rmse.append(metrics['rmse'])
            test_nll_loss.append(metrics['nll_loss'])
            test_mse_loss.append(metrics['mse_loss'])

            test_mape_list.append(metrics['mape_list'])
            test_rmse_list.append(metrics['rmse_list'])
            test_mae_list.append(metrics['mae_list'])


        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_nll_loss = np.mean(train_nll_loss)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_nll_loss = np.mean(valid_nll_loss)

        mtest_loss = np.mean(test_loss)
        mtest_mape = np.mean(test_mape)
        mtest_rmse = np.mean(test_rmse)
        mtest_nll_loss = np.mean(test_nll_loss)

        mtest_mape_list = np.mean(np.array(test_mape_list), 0)
        mtest_rmse_list = np.mean(np.array(test_rmse_list), 0)
        mtest_mae_list = np.mean(np.array(test_mae_list), 0)

        his_loss.append(mvalid_loss)

        wandb.log({
            'time/train_time': train_time[-1],
            'time/val_time': val_time[-1],
            'loss/train_loss': mtrain_loss,
            'loss/val_loss': mvalid_loss,
            'errors/train_mape': mtrain_mape,
            'errors/train_rmse': mtrain_rmse,
            'errors/val_mape': mvalid_mape,
            'errors/val_rmse': mvalid_rmse,
            'errors/test_mape': mtest_mape,
            'errors/test_rmse': mtest_rmse,
            'loss/train_nll_loss': mtrain_nll_loss,
            'loss/val_nll_loss': mvalid_nll_loss,
            'loss/test_nll_loss': mtest_nll_loss,
            'loss/train_mse_loss': np.mean(train_mse_loss),
            'loss/val_mse_loss': np.mean(valid_mse_loss),
            'loss/test_mse_loss': np.mean(test_mse_loss),
        }, step=i)

        # for j in range(len(args.pred_len)):
        for j in [2, 5, 8, 11]:
            wandb.log({
                f'errors_spec/test_mape_{j}': mtest_mape_list[j],
                f'errors_spec/test_rmse_{j}': mtest_rmse_list[j],
                f'errors_spec/test_mae_{j}': mtest_mae_list[j],
            }, step=i)

        # engine.summary.add_scalar('loss/rho', torch.sigmoid(engine.covariance.rho).item(), i)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)

        if i % args.save_every == 0:
            # torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss, 2))+".pth")
            if best_val_loss > mvalid_mape:
                best_val_loss = mvalid_mape
                engine.save(best=True)
                print("Saved best model")

            else:
                engine.save()
                print("Saved model")

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    engine.load(model_path=f'{engine.logdir}/best_model_list.pt')

    test_loss = []
    test_mape = []
    test_rmse = []
    test_nll_loss = []
    test_crps_loss = []
    test_es_loss = []

    s1 = time.time()
    for iter, (x, y) in enumerate(dataloader['test_loader']):
        testx = x.to(device)
        testx = testx.transpose(1, 3)
        testy = y.to(device)
        testy = testy.transpose(1, 3)

        metrics = engine.eval(testx, testy[:, 0, :, :], crps=True)

        test_loss.append(metrics['loss'])
        test_mape.append(metrics['mape'])
        test_rmse.append(metrics['rmse'])
        test_nll_loss.append(metrics['nll_loss'])
        test_crps_loss.append(metrics['crps'])
        test_es_loss.append(metrics['ES'])

    mtest_loss = np.mean(test_loss)
    mtest_mape = np.mean(test_mape)
    mtest_rmse = np.mean(test_rmse)
    mtest_nll_loss = np.mean(test_nll_loss)
    mtest_crps_loss = np.mean(test_crps_loss)
    mtest_es_loss = np.mean(test_es_loss)

    print("Testing Results:")
    print("Test Loss: {:.4f}".format(mtest_loss))
    print("Test MAPE: {:.4f}".format(mtest_mape))
    print("Test RMSE: {:.4f}".format(mtest_rmse))
    print("Test NLL Loss: {:.4f}".format(mtest_nll_loss))
    print("Test CRPS Loss: {:.4f}".format(mtest_crps_loss))
    print("Test ES Loss: {:.4f}".format(mtest_es_loss))

    wandb.log({
        'time/test_time': time.time() - s1,
        'test/test_loss': mtest_loss,
        'test/test_mape': mtest_mape,
        'test/test_rmse': mtest_rmse,
        'test/test_nll_loss': mtest_nll_loss,
        'test/test_crps_loss': mtest_crps_loss,
        'test/test_es_loss': mtest_es_loss,
    })



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
