import torch
import torch.nn as nn
import math
import copy
from typing import Tuple, Optional, List, Callable
import abc
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import pathlib
import numpy as np

import sys, os

os.environ["PATH"] = os.environ["PATH"] + ":" + os.path.realpath("./third_party/mbrl")

from mbrl.models import Model

# 30只股票一起放进去    
class ModelPredict(Model):
    def __init__(self, teacher_ratio=0.2, ensemble_nums=5, update_window=30):
        super().__init__()
        self.input_dim = 30
        self.ensemble_nums = ensemble_nums
        self.update_window = update_window
        self.register_buffer("indexes", torch.zeros(self.input_dim, dtype=torch.long))
        # self.teacher_ratio = teacher_ratio
        lstm_func = lambda: nn.LSTM(
            input_size=2,
            hidden_size=32,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
        )
        mlp_func = lambda: nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        self.lstms = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for _ in range(ensemble_nums):
            self.lstms.append(lstm_func())
            self.mlps.append(mlp_func())

        self.loss_func = nn.MSELoss(reduction="none")
    
    #加入(x_n+1 - x_n)为特征,进行forward
    def forward(
        self,
        history_price_seq: torch.tensor,
        actions,#每日单只股票的操作序列
        vol,#每日单只股票交易量
    ):
        history_price_seq = history_price_seq[
            :, max(0, history_price_seq.shape[1] - self.update_window) :
        ]
        #delta_price_seq  <-->  y_n = (x_n+1 - x_n)
        delta_price_seq = history_price_seq[:,1:] - history_price_seq[:,:-1]
        pad = torch.zeros((self.stock_dim,1),device = history_price_seq.device)
        delta_price_seq = torch.cat([pad,delta_price_seq],dim = 1)

        #加入log(action)/log(vol),表示每一操作对价格的影响，进而影响s_n+1
        action_inful = actions.log()/vol.log()

        #lstm input = (history_price_seq, delta_price_seq)
        lstm_input_seq = torch.cat([history_price_seq.unsqueeze(2),delta_price_seq.unsqueeze(2)],dim = 2)

        prices = torch.zeros(
            (history_price_seq.shape[0], self.ensemble_nums),
            device=history_price_seq.device,
        )
        
        for i, (lstm, mlp) in enumerate(zip(self.lstms, self.mlps)):
            _, (h, _) = lstm(lstm_input_seq)
            mlp_input = torch.cat([h[-1],action_inful.unsqueeze(1)],dim = 1)
            prices[:, i] = mlp(mlp_input).squeeze()
        return prices

    def loss(
        self,
        model_input: torch.tensor,
        target: Optional[torch.Tensor] = None,
    ):
        prices = self.forward(model_input, None)
        mse = self.loss_func(prices, target.unsqueeze(1).repeat(1, self.ensemble_nums))
        return prices, mse

    @torch.no_grad()
    def eval_score(
        self, model_input: torch.tensor, target: Optional[torch.Tensor] = None
    ):
        """
        return：最大误差，最大误差百分比，最小误差，最小误差百分比，平均误差百分比,mse误差
        return(
            max((predict_price)-true_price)/true_price %),
            min((predict_price)-true_price)/true_price %),
            mean((predict_price)-true_price)/true_price %)
        """
        next_price = self.inference(model_input)
        error = abs((next_price.squeeze() - target) / target)
        max_error_percent, _ = error.topk(1)
        error = -1 * error
        min_error_percent, _ = error.topk(1)
        mean_error_percent = -1 * error.mean()
        error_mse = self.loss_func(next_price.squeeze(), target)
        return (
            next_price,
            max_error_percent,
            -1 * min_error_percent,
            mean_error_percent,
            error_mse.mean(),
        )

    def update_indexes(self, indexes):
        self.indexes = indexes

    def inference(self, history_price_seq: torch.tensor, actions, vol):
        history_price_seq = history_price_seq[:, max(0, history_price_seq.shape[1] - self.update_window) :]

        delta_price_seq = history_price_seq[:,1:] - history_price_seq[:,:-1]
        pad = torch.zeros((self.stock_dim,1),device = history_price_seq.device)
        delta_price_seq = torch.cat([pad,delta_price_seq],dim = 1)

        action_inful = actions.log()/vol.log()

        lstm_input_seq = torch.cat([history_price_seq.unsqueeze(2),delta_price_seq.unsqueeze(2)],dim = 2)
        
        output_tensor = torch.zeros((self.input_dim, 1), device=history_price_seq.device)
        """按照lstm_i（ensembel_number）对股票进行分块,将同一模型处理的股票同时计算"""
        index_arg = self.indexes.argsort()  # argsort(）返回 索引：处理对象 从小到大
        position = 0
        for i in range(self.ensemble_nums):
            num_i = (self.indexes == i).sum()
            if num_i > 0:
                indexes_i = index_arg[position : num_i + position]
                position += num_i
                lstm_input_seq_i = lstm_input_seq[indexes_i]
                _, (h, _) = self.lstms[i](lstm_input_seq_i)
                action_influ_i = action_inful.unsqueeze(1)[indexes_i]
                mlp_input_i = torch.cat([h[-1], action_influ_i],dim = 1)
                output_tensor[indexes_i] = self.mlps[i](mlp_input_i)
        return output_tensor


def train(
    model: ModelPredict,
    dataset_train,
    dataset_val,
    actions,
    trade_vol,
    num_epochs,
    device,
    teacher_forcing = 0.2,
    path: pathlib.Path = "",
    callback_train: Callable = None,
    callback_val: Callable = None,
):
    epoch_iter = range(num_epochs)

    optim_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr)

    model.to(device)
    for epoch in epoch_iter:
        print(f"epoch ======================={epoch}==========================")
        mses = torch.zeros((model.input_dim, model.ensemble_nums), device=device)
        predict_price: List = []
        for day in range(1, dataset_train.shape[1]):  # dataset_train.shape[1]
            input = dataset_train[:, :day]
            # 加入teacher forcing
            if day > 1 and random.random() < teacher_forcing:
                input[:, -1] = predict_price[-1].squeeze()
            target = dataset_train[:, day]
            input, target = input.to(device), target.to(device)
            model.train()
            optimizer.zero_grad()
            # 找到30只股票的历史mse之和最小的为其预测index
            prices, mse = model.loss(input, target, actions, trade_vol)
            mses = mses + mse
            indexes = mses.argmin(-1)
            model.update_indexes(indexes)
            # 索引出预测价格
            price = prices.gather(1, indexes.unsqueeze(1))
            predict_price.append(price.clone().detach())
            mse.mean().backward()
            optimizer.step()
            print(mse.detach().mean().item())
            if callback_train is not None:
                callback_train(epoch, day, mse.detach().cpu().mean().numpy())
        model.save(path)

        res = torch.zeros((dataset_val.shape[1], 4))
        next_prices = torch.zeros((dataset_val.shape[0], dataset_val.shape[1]))
        for day in range(0, dataset_val.shape[1]):
            input =  torch.cat([dataset_train,dataset_val[:, :day]], dim = 1)
            target = dataset_val[:, day]
            input, target = input.to(device), target.to(device)
            (
                next_prices[:, day:day+1],
                res[day, 0],
                res[day, 1],
                res[day, 2],
                res[day, 3],
            ) = model.eval_score(input, target, actions, )
            val_loss_set = res[day].detach().cpu().numpy()
            if callback_val is not None:
                callback_val(epoch, day, next_prices[:, day].detach().cpu().numpy(), val_loss_set)
        print(
            f"max_error_percent = {res[:,0].max()}, min_error_percent = {res[:,1].min()}, mean_error_percent = {res[:,2].mean()}, error_mse = {res[:,3].mean()}"
        )
        



# train_loss val_loss  plot

# 5 个LSTM同时跑？
# RL daily update price 不更新参数 ;p_n+1 = model(p_n); 30个(s_n,a,s_n+1,r),buffer_update:当新放入30个四元组，就讲原有的最旧的30个四元组排除; 4000（一只股票 rolling 4000 daily）*30 = buffer_size
# If epoch/30 ==0,train a ensemble lstm ,每只股票给一个ensemble lstm的index，同时五个model save；dynamic_model 30 update
