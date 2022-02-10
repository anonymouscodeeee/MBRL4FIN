import pandas as pd
import numpy as np
from datetime import datetime, timedelta  # 记录outputs_time，记录循环用时
import matplotlib as mpl
mpl.rcParams.update({"font.size": 16})
import matplotlib.pyplot as plt


# matplotlib.use('Agg')
from IPython import display
#%matplotlib inline
import torch
from pprint import pprint
import itertools
import omegaconf
import pathlib
import random


from stable_baselines3 import A2C

# import sys
# sys.path.append("../FinRL-Library")

# import sys,os
# sys.path.append(os.path.dirname(os.path.realpath(".")))
import sys, os

from finrl.config import config  # 引入finrl包的配置
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.model.models import DRLAgent, DRLEnsembleAgent



import third_party.mbrl.mbrl.planning as planning
from env_stocktrading import StockTradingEnv
from dynamic_env import DynamicEnv
import data_process
import config
import plot

#%load_ext autoreload
#%autoreload 

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"



##########提取预处理之后的数据
processed_full, df_train, df_val = data_process.preprocess(
    config.START_DATE,
    config.END_DATE,
    config.DOW_30_TICKER,
    config.TRAIN_START,
    config.TRAIN_END,
    config.VAL_TEST_START,
    config.VAL_TEST_END,
)

# 将原始close_price_seq 输入给 dynamic_model
list_ticker = processed_full["tic"].unique().tolist()
list_ticker.sort()
print(list_ticker)

index_price_train = data_process.data_extra(df_train, "close", list_ticker)
trade_vol_train = data_process.data_extra(df_train, "volume", list_ticker)

index_price_val = data_process.data_extra(df_val, "close", list_ticker)
trade_vol_val = data_process.data_extra(df_val, "volume", list_ticker)

train_size = index_price_train.shape[1]
val_size = index_price_val.shape[1]

train_data = index_price_train  # + trade_vol_train
val_data = index_price_val  # + trade_vol_val

print(train_data.shape)
print(val_data.shape)


import dynamic_model

model_predict = dynamic_model.ModelPredict()
model_predict.to(device=DEVICE)
# model_trainer = models.ModelTrainer(model_predict, optim_lr=1e-3, weight_decay=5e-5)



stock_dimension = len(processed_full.tic.unique())
state_space = 1 + 2 * stock_dimension + len(config.TECH_INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
print(config.TECH_INDICATORS)
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.TECH_INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity": 5,
    "day": train_size,
}

#########true env
seed = 0
env = StockTradingEnv(
    processed_full,
    **env_kwargs,
    iteration=10,
)
env.seed(seed)


# 测试 env处理 随机action
# import random
# action = np.array([random.uniform(-1,1) for i in range(30)])
# state, reward, terminal =env.step(action)


# 加入load,从path 加载模型
PATH = pathlib.Path("./checkpoint/dynamic_model_paras")
if PATH.is_file():
    model_predict.load(PATH)

# 使用callback 将train的loss data保存
num_epochs = 5
stock_dim = 30
callback_train_matrix: np.array = np.zeros((num_epochs, train_size - 1))


def callback_train(epoch, day, mse):
    callback_train_matrix[epoch, day - 1] = mse


callback_val_matrix: np.array = np.zeros((num_epochs, val_size, 4))
close_prices = np.zeros(
    (
        num_epochs,
        val_size,
        stock_dim,
    )
)


def callback_val(epoch, day, close_price, val_loss_set):
    close_prices[epoch, day] = close_price
    callback_val_matrix[epoch, day] = val_loss_set


# 调用train函数训练dynamic_model
"""
dynamic_model.train(
    model_predict,
    train_data,
    val_data,
    trade_vol,
    num_epochs=num_epochs,
    device = DEVICE,
    teacher_forcing=0.2,
    path=PATH,
    callback_train=callback_train,
    callback_val=callback_val,
)
"""

plot_list = [2, 3, 6, 8, 9, 25]
plot_names = [list_ticker[i] for i in plot_list]
# fig, axes = plt.subplots(2,3)
x = list(range(200))
for i in range(2):
    for j in range(3):
        plt.figure(3 * i + j)
        # plt.plot(x,close_prices[-1, 500:500+200, plot_list[3*i+j]])
        plt.plot(x, val_data[plot_list[3 * i + j], :200].cpu())
        plt.title(plot_names[3 * i + j])
plt.show()


agent_cfg = omegaconf.OmegaConf.create(
    {
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 5,
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 3,
            "elite_ratio": 0.1,
            "population_size": 800,
            "alpha": 0.1,
            "device": DEVICE,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True,
        },
    }
)


stock_dim = 30
state = [1000000] + train_data[:, -1].tolist() + stock_dim * [0]

model_env = DynamicEnv(
    model_predict,
    history_price_seq=train_data,  # n*stock_dim
    date=train_size,
    stock_dim=stock_dim,
    hmax=100,
    initial_amount=1000000,
    buy_cost_pct=0.001,
    sell_cost_pct=0.001,
    reward_scaling=1e-4,
    state_space=61,
    action_space=30,
    make_plots=False,
    print_verbosity=10,
    initial=False,
    previous_state=state,
    model_name="",
    mode="",
    iteration="",
    use_real_price=True,
    true_price_seq=val_data,
)
agent = planning.create_trajectory_optim_agent_for_model(
    model_env,
    agent_cfg,
    num_particles=1,  # 一般RL中，采取action会得到不同的s_n+1，故使用num_particles来降低不确定性，但此处同一action会得到相同s_n+1，故取num_particles = 1
)

# Main PETS loop
# 引入时间，方便记录每次的outputs
time_current = datetime.now() + timedelta(hours=8)
time_current = time_current.strftime("%Y-%m-%d_%H:%M")

csv_name_func = lambda time, i: f"outputs/pets_{time}_{i}_account_value.csv"

date_val = df_val["date"].unique().tolist()
date_val.sort()

trial_length = val_size  # val_size
num_trials = 3
all_rewards = [0]
account_value = []


for trial in range(num_trials):
    starttime = datetime.now()  # 计算每个trail时间
    seed = trial
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    account_value.append([])
    state = torch.tensor(env.reset(train_size)[0:61], device=DEVICE)
    history_price_seq = train_data
    agent.reset()  # env.reset()

    done = False
    total_reward = 0.0
    steps_trial = 0
    day = 0

    while not done:
        obs = {"history_price_seq": history_price_seq, "initial_state": state}
        action = agent.act(obs)
        states, reward, terminal = env.step(action)

        state = torch.tensor(states[0:61], device=DEVICE)
        history_price_seq = torch.cat([history_price_seq, val_data[:, 0:day]], dim=1)

        account_value[-1].append(env.calc_asset())
        print(f"{date_val[steps_trial]} account_value===={account_value[-1][-1]}====")
        day += 1
        total_reward += reward
        steps_trial += 1

        if steps_trial == trial_length:
            break

    all_rewards.append(total_reward)
    endtime = datetime.now()
    print(f"trial {trial} duation ======{(endtime - starttime).seconds}======")
    df_account_value = pd.DataFrame(
        {"account_value": account_value[-1], "date": date_val[:trial_length]}
    )
    df_account_value["daily_return"] = (
        df_account_value["account_value"].pct_change().tolist()
    )
    df_account_value["datadate"] = date_val[:trial_length]
    df_account_value.to_csv(csv_name_func(time_current, trial))


#%matplotlib inline
# 选择想要画图的数据时间
#time_plot = "2021-08-27_16:29"
time_plot = time_current
for trial in range(num_trials):
    df_account_value_trial = pd.read_csv(csv_name_func(time_plot, trial))
    plot.account_value_show(df_account_value_trial)

# train_loss val_loss  plot
# x-epoch ;y_1 = the mean mse of all epoch for train_data, y_2 = the mean mse of all epoch for val_data
# asset plot
# x-epoch ;y-final asset for all epoch
