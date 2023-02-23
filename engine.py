import math
import torch.optim as optim
from model import *
import util
import torch.nn as nn
from better_loss import covariance, batch_opt


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate,
                 wdecay, device, supports, gcn_bool, addaptadj, aptinit, delay,
                 train_L_space, train_L_time, train_L_batch, rho, det, nll):

        self.seq_length = seq_length
        if nll == "GAL":
            out_dim = seq_length * 2
        else:
            out_dim = seq_length
        self.nll = nll

        self.model = gwnet(device, num_nodes, dropout,
                           supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                           aptinit=aptinit, in_dim=in_dim, out_dim=out_dim,
                           residual_channels=nhid, dilation_channels=nhid,
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.covariance = covariance(num_nodes, delay, seq_length, device,
                                     train_L_space=train_L_space,
                                     train_L_time=train_L_time,
                                     train_L_batch=train_L_batch)

        # self.loss = util.masked_mae
        self.loss_class = batch_opt(num_nodes, delay, rho=rho, det=det, nll=nll)
        self.loss = self.loss_class.loss

        self.model_list = nn.ModuleDict(
            {
                'model': self.model,
                'covariance': self.covariance
            }
        )

        self.optimizer = optim.Adam(self.model_list.parameters(), lr=lrate, weight_decay=wdecay)

        self.scaler = scaler
        self.clip = 5

        self.act = nn.Softplus()

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        loss, predict = self.process_batch(input, real_val)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape, rmse, mae = self.get_metrics(predict, real_val)

        return loss.item(), mape, rmse, mae

    def get_metrics(self, predict, real_val):
        if self.nll == "GAL":
            predict = predict[..., :self.seq_length]

        mape = util.masked_mape(predict, real_val, 0).item()
        rmse = util.masked_rmse(predict, real_val, 0).item()
        mae = util.masked_mae(predict, real_val, 0).item()

        return mape, rmse, mae

    def process_batch(self, input, real_val, test=False):
        b, d, f, n, t = input.shape
        input = input.reshape(b*d, f, n, t)

        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        mask = real_val != 0.0

        L_list = self.covariance.get_L()

        predict = predict.reshape(b, d, predict.shape[2], predict.shape[3])
        if test:
            if self.nll == "GAL":
                predict = predict[..., :self.seq_length]
            return predict.detach()

        loss = self.loss(predict, real, L_list, mask).mean()
        return loss, predict.detach()

    def eval(self, input, real_val):
        self.model.eval()

        with torch.no_grad():
            predict = self.process_batch(input, real_val, test=True)
            mape, rmse, mae = self.get_metrics(predict, real_val)
            return 0, mape, rmse, mae
