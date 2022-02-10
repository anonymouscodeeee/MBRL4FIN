import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from pathlib import Path

COST = 0.001
STOCK_DIM = 30


def sell_stock(index, obs, action):
    def _do_sell_normal(state):
        if state[index + 1] > 0:
            # Sell only if the price is > 0 (no missing data in this particular date)
            # perform sell action based on the sign of the action
            if state[index + STOCK_DIM + 1] > 0:
                # Sell only if current asset is > 0
                sell_num_shares = min(abs(action), state[index + STOCK_DIM + 1])
                sell_amount = state[index + 1] * sell_num_shares * (1 - COST)
                # update balance
                state[0] += sell_amount

                state[index + STOCK_DIM + 1] -= sell_num_shares
            else:
                sell_num_shares = 0
        else:
            sell_num_shares = 0

        return sell_num_shares

    # perform sell action based on the sign of the action
    sell_num_shares = _do_sell_normal(obs)

    return sell_num_shares, obs


def buy_stock(index, obs, action):
    def _do_buy(state):
        if state[index + 1] > 0:
            # Buy only if the price is > 0 (no missing data in this particular date)
            available_amount = state[0] // state[index + 1]  # //整除，得到最大可以买入的数量
            # print('available_amount:{}'.format(available_amount))

            # update balance
            buy_num_shares = min(available_amount, action)
            buy_amount = state[index + 1] * buy_num_shares * (1 + COST)
            state[0] -= buy_amount

            state[index + STOCK_DIM + 1] += buy_num_shares
        else:
            buy_num_shares = 0

        return buy_num_shares

    # perform buy action based on the sign of the action
    buy_num_shares = _do_buy(obs)

    return buy_num_shares, obs


# rollout 分析
def draw(data, output_dir):
    print(f"rollout times: {len(data)}, keys:")
    for key, _ in data[0].items():
        print(key)
        
    print(f"rollout times: {len(data)}, keys:")
    for key, _ in data[0].items():
        print(key)
    # 为了观察结果随时间的变化，先把所有rollout的对应结果cat到一起
    # 结果应该为[rollout, batch, feature]
    all_actions = np.expand_dims(data[0]["actions"], 0)
    all_obs = np.expand_dims(data[0]["obs"], 0)
    all_next_obs = np.expand_dims(data[0]["next_obs"], 0)
    all_reward = np.expand_dims(data[0]["next_rewards"], 0)
    for i in range(len(data) - 1):
        all_actions = np.concatenate(
            [all_actions, np.expand_dims(data[i + 1]["actions"], 0)], axis=0
        )
        all_obs = np.concatenate(
            [all_next_obs, np.expand_dims(data[i + 1]["obs"], 0)], axis=0
        )
        all_next_obs = np.concatenate(
            [all_next_obs, np.expand_dims(data[i + 1]["next_obs"], 0)], axis=0
        )
        all_reward = np.concatenate(
            [all_reward, np.expand_dims(data[i + 1]["next_rewards"], 0)], axis=0
        )

    ################# 一.observation 分析 ################
    ######1.shares也是可以通过obs+action准确计算的
    ######2.balance转换关系，正确的balance应该是可以通过obs+action 准确计算得到的
    cache_path = output_dir / "all_obs_true.npz"
    if cache_path.is_file():
        all_obs_true = np.load(cache_path)
    else:
        all_obs_true = np.zeros_like(all_obs)
        for rollout in range(len(data)):
            for batch in range(len(data[rollout]["actions"])):
                obs_true = all_obs[rollout][batch].copy()
                actions = all_actions[rollout][batch]
                argsort_actions = np.argsort(actions)
                sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
                buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]
                for index in sell_index:
                    sell_stock(index, obs_true, actions[index] * 100)
                for index in buy_index:
                    buy_stock(index, obs_true, actions[index] * 100)
                all_obs_true[rollout, batch] = obs_true
        np.savez(cache_path, all_obs_true=all_obs_true)

    # 比较计算得到的和模型预测的obs中blance和shares的差别
    obs_odd = all_next_obs - all_obs_true
    shares_mean_odd = np.abs(obs_odd[:, :, 31:61]).mean(axis=-1).mean(-1)
    plt.figure(0)
    plt.plot(shares_mean_odd)
    plt.xlabel("Iteration", fontsize=14)
    plt.tick_params(labelsize=8)
    plt.ylabel("Absolute error", fontsize=14)
    plt.title("")
    plt.savefig(output_dir / "predicted_shares_error.pdf", format="pdf")
    balace_odd = np.abs(obs_odd[:, :, 0]).mean(-1)
    plt.figure(1)
    plt.xlabel("Iteration", fontsize=14)
    plt.tick_params(labelsize=8)
    plt.ylabel("Absolute error", fontsize=14)
    plt.title("")
    plt.plot(balace_odd)
    plt.savefig(output_dir / "predicted_balance_error.pdf", format="pdf")

######3.close价格预测
######4.各种指标
################# 二.rewards 分析 #################
# 正确的rewards应该是可以根据obs+next_obs一起计算出来的

# epoch(episode) - actor loss and critic loss

if __name__ == "__main__":
    with open(
    "outputs/2022-02-09/mbpo_sp/mbpo_8000/mbpo_stats.pkl",
    "rb",
    ) as f:
        unpickler = pickle.Unpickler(f)
        data = unpickler.load()
        
    