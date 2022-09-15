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

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
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
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
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
parser.add_argument("--mse_coef", type=float, default=0.1)
parser.add_argument("--flow", action="store_true")
parser.add_argument('--nonlinearity', type=str, default='softmax', choices=["softmax", "softplus", "elu", "sigmoid", "exp"])

args = parser.parse_args()

args.pred_len = [2, 5, 8, 11]


def main():
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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

    # target_sensors = sensor_ids

    target_sensor_inds = [sensor_id_to_ind[i] for i in target_sensors]
    args.num_nodes = len(target_sensors)

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size,
                                   target_sensor_inds=target_sensor_inds, flow=args.flow)
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

    # A = (adjinit != 0).float()
    # A[torch.arange(A.shape[0]), torch.arange(A.shape[0])] = 0
    # D = A.sum(0)
    # L = A - torch.diag(D)
    # L = L.float()

    # # eigenvalues of L
    # eigvals, eigvecs = torch.eig(L, eigenvectors=True)

    # engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
    #                  args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
    #                  adjinit)

    # engine = MDN_trainer(scaler, args.in_dim,
    #                      args.seq_length,
    #                      args.num_nodes,
    #                      args.num_rank,
    #                      args.nhid,
    #                      args.dropout,
    #                      args.learning_rate,
    #                      args.weight_decay,
    #                      device, supports,
    #                      args.gcn_bool,
    #                      args.addaptadj,
    #                      adjinit,
    #                      n_components=args.n_components,
    #                      reg_coef=args.reg_coef,
    #                      pred_len=args.pred_len)

    engine = MDN_trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.num_rank, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, n_components=args.n_components, reg_coef=args.reg_coef, consider_neighbors=args.consider_neighbors,
                         outlier_distribution=args.outlier_distribution, pred_len=args.pred_len, rho=args.rho, diag=args.diag,
                         mse_coef=args.mse_coef, nonlinearity=args.nonlinearity)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    best_val_loss = float('inf')
    for i in range(args.epochs):
        # if i % 10 == 0:
        #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # for g in engine.optimizer.param_groups:
        #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        train_mse_loss = []
        train_nll_loss = []
        train_reg_loss = []
        # train_crps_loss = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics['loss'])
            train_mape.append(metrics['mape'])
            train_rmse.append(metrics['rmse'])
            train_nll_loss.append(metrics['nll_loss'])
            train_reg_loss.append(metrics['reg_loss'])
            train_mse_loss.append(metrics['mse_loss'])
            # train_crps_loss.append(metrics["crps"])

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
        valid_reg_loss = []
        valid_mse_loss = []
        valid_crps_loss = []
        valid_es_loss = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)

            metrics = engine.eval(testx, testy[:, 0, :, :])

            valid_loss.append(metrics['loss'])
            valid_mape.append(metrics['mape'])
            valid_rmse.append(metrics['rmse'])
            valid_nll_loss.append(metrics['nll_loss'])
            valid_reg_loss.append(metrics['reg_loss'])
            valid_mse_loss.append(metrics['mse_loss'])
            valid_crps_loss.append(metrics["crps"])
            valid_es_loss.append(metrics["ES"])

            if i % 1 == 0:
                if iter == 0:
                    engine.plot_cov()

        test_loss = []
        test_mape = []
        test_rmse = []
        test_nll_loss = []
        test_reg_loss = []
        test_mse_loss = []
        test_crps_loss = []
        test_es_loss = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)

            metrics = engine.eval(testx, testy[:, 0, :, :])

            test_loss.append(metrics['loss'])
            test_mape.append(metrics['mape'])
            test_rmse.append(metrics['rmse'])
            test_nll_loss.append(metrics['nll_loss'])
            test_reg_loss.append(metrics['reg_loss'])
            test_mse_loss.append(metrics['mse_loss'])
            test_crps_loss.append(metrics["crps"])
            test_es_loss.append(metrics["ES"])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_nll_loss = np.mean(train_nll_loss)
        mtrain_reg_loss = np.mean(train_reg_loss)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_nll_loss = np.mean(valid_nll_loss)
        mvalid_reg_loss = np.mean(valid_reg_loss)
        mvalid_crps_loss = np.mean(valid_crps_loss)
        mvalid_es_loss = np.mean(valid_es_loss)

        his_loss.append(mvalid_loss)

        mtest_loss = np.mean(test_loss)
        mtest_mape = np.mean(test_mape)
        mtest_rmse = np.mean(test_rmse)
        mtest_nll_loss = np.mean(test_nll_loss)
        mtest_reg_loss = np.mean(test_reg_loss)
        mtest_crps_loss = np.mean(test_crps_loss)
        mtest_es_loss = np.mean(test_es_loss)

        his_loss.append(mvalid_loss)

        engine.summary.add_scalar('time/train_time', train_time[-1], i)
        engine.summary.add_scalar('time/val_time', val_time[-1], i)

        engine.summary.add_scalar('loss/train_loss', mtrain_loss, i)
        engine.summary.add_scalar('loss/val_loss', mvalid_loss, i)

        engine.summary.add_scalar('errors/train_mape', mtrain_mape, i)
        engine.summary.add_scalar('errors/train_rmse', mtrain_rmse, i)
        # engine.summary.add_scalar('errors/train_crps', mtrain_crps_loss, i)
        engine.summary.add_scalar('errors/val_mape', mvalid_mape, i)
        engine.summary.add_scalar('errors/val_rmse', mvalid_rmse, i)
        engine.summary.add_scalar('errors/val_crps', mvalid_crps_loss, i)
        engine.summary.add_scalar('errors/val_es', mvalid_es_loss, i)

        engine.summary.add_scalar('errors/test_mape', mtest_mape, i)
        engine.summary.add_scalar('errors/test_rmse', mtest_rmse, i)
        engine.summary.add_scalar('errors/test_crps', mtest_crps_loss, i)
        engine.summary.add_scalar('errors/test_es', mtest_es_loss, i)

        engine.summary.add_scalar('loss/train_nll_loss', mtrain_nll_loss, i)
        engine.summary.add_scalar('loss/train_reg_loss', mtrain_reg_loss, i)
        engine.summary.add_scalar('loss/val_nll_loss', mvalid_nll_loss, i)
        engine.summary.add_scalar('loss/val_reg_loss', mvalid_reg_loss, i)
        engine.summary.add_scalar('loss/test_nll_loss', mtest_nll_loss, i)
        engine.summary.add_scalar('loss/test_reg_loss', mtest_reg_loss, i)
        engine.summary.add_scalar('loss/train_mse_loss', np.mean(train_mse_loss), i)
        engine.summary.add_scalar('loss/val_mse_loss', np.mean(valid_mse_loss), i)
        engine.summary.add_scalar('loss/test_mse_loss', np.mean(test_mse_loss), i)

        # engine.summary.add_scalar('loss/rho', torch.sigmoid(engine.covariance.rho).item(), i)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)

        if i % args.save_every == 0:
            # torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss, 2))+".pth")
            if best_val_loss > mvalid_es_loss:
                best_val_loss = mvalid_es_loss
                engine.save(best=True)
                print(f"Saved best model at epoch {i} with loss {best_val_loss}")

            else:
                engine.save()
                print("Saved model")

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    engine.load(model_path=f'{engine.logdir}/best_model.pt',
                cov_path=f'{engine.logdir}/best_covariance.pt',
                fc_w_path=f'{engine.logdir}/best_fc_w.pt')

    test_loss = []
    test_mape = []
    test_rmse = []
    test_nll_loss = []
    test_reg_loss = []
    test_mse_loss = []
    test_crps_loss = []

    s1 = time.time()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)

        metrics = engine.eval(testx, testy[:, 0, :, :])

        test_loss.append(metrics['loss'])
        test_mape.append(metrics['mape'])
        test_rmse.append(metrics['rmse'])
        test_nll_loss.append(metrics['nll_loss'])
        test_reg_loss.append(metrics['reg_loss'])
        test_mse_loss.append(metrics['mse_loss'])
        test_crps_loss.append(metrics["crps"])

    mtest_loss = np.mean(test_loss)
    mtest_mape = np.mean(test_mape)
    mtest_rmse = np.mean(test_rmse)
    mtest_nll_loss = np.mean(test_nll_loss)
    mtest_reg_loss = np.mean(test_reg_loss)
    mtest_mse_loss = np.mean(test_mse_loss)
    mtest_crps_loss = np.mean(test_crps_loss)

    print("Testing Results:")
    print("Test Loss: {:.4f}".format(mtest_loss))
    print("Test MAPE: {:.4f}".format(mtest_mape))
    print("Test RMSE: {:.4f}".format(mtest_rmse))
    print("Test NLL Loss: {:.4f}".format(mtest_nll_loss))
    print("Test Reg Loss: {:.4f}".format(mtest_reg_loss))
    print("Test MSE Loss: {:.4f}".format(mtest_mse_loss))
    print("Test CRPS Loss: {:.4f}".format(mtest_crps_loss))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
