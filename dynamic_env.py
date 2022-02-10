from typing import Dict
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common import logger
from dynamic_model import ModelPredict


class DynamicEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        model_predict:ModelPredict,
        history_price_seq: torch.tensor,
        date,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        make_plots=False,
        print_verbosity=10,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        day = 0,
        use_real_price: bool = False,
        true_price_seq = None,
    ):
        self.model_predict:ModelPredict = model_predict
        self.history_price_seq = history_price_seq
        self.date = date
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling

        self.action_space = action_space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))

        self.state_space = state_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )

        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # initialize reward
        self.reward = 0
        self.episode = 0

        # initalize state
        self.init()
        self._seed()

        #initial close_price true or false
        self.use_real_price = use_real_price
        self.true_price_seq = true_price_seq

    # 定义交易函数，包含buy和sell函数，依照（-1,1）分别进行交易
    def trade(self, index, action):
        #若预测股票价格<0,则将其交易数量变为零
        prices_nomal = self.state[:, 1 + index].detach().cpu()
        prices_nomal.apply_(lambda x:1 if x > 0 else 0)
        prices_nomal = prices_nomal.to(device = self.state.device)
        tread_shares = self.state[:, 1 + index + self.stock_dim] * prices_nomal

        sell_index = action < 0
        tread_shares[sell_index] = torch.min(tread_shares[sell_index], action[sell_index].abs()) * (-1)

        buy_index = action > 0
        available_amount = self.state[:,0] // self.state[:,index + 1]
        tread_shares[buy_index] = torch.min(available_amount[buy_index], action[buy_index])
        
        self.state[:,0][sell_index] += (tread_shares[sell_index] * (-1))* self.state[:,1+index][sell_index] * (1 - self.sell_cost_pct)
        self.state[:,0][buy_index] += (tread_shares[buy_index] * (-1))* self.state[:,1+index][buy_index] * (1 + self.buy_cost_pct)
        self.state[:, 1 + index + self.stock_dim] += tread_shares

        return tread_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig("results/account_value_trade_{}.png".format(self.episode))
        plt.close()

    def step(self, actions):
        # self.date = date

        self.terminal = torch.zeros_like(actions)# How?设置一个balance阈值，比如：止损下线，90%initial; 只抛不买 if 预判采取action之后asset<90  else 执行action
        if False and self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = end_total_asset - self.initial_amount

            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades_times}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record(
            #    "environment/total_reward_pct",
            #    (tot_reward / (end_total_asset - tot_reward)) * 100,
            # )
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades_times)

            return self.state, self.reward, self.terminal, {}

        else:
            # 对(n_batch,stock_dim)大小的action Tensor，按照 行(n_batch) argsort 得到 sell\buy 的股票的 index，同时按index顺序索引 close_price(30),shares(30)
            # 现已保证每一行的 action 和 price、shares 正确对应
            # 按照 列(30) 对action的正负 给出相应的（1，-1）列向量; -1:min(abs(action),shares), 1:min(abs(action),h_max = 100)
            # 按照action顺序更新shares； 更新balance；给出reward(n_batch个)；     注意：此时state(index顺序)
            # 按照原index顺序，将state(index顺序)，还原为原有的state顺序

            # 函数evaluate_action_sequences，调用step，由上述reward指导选出相应的action(30只股票)

            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.to(
                dtype=int
            )  # convert into integer because we can't by fraction of shares

            begin_total_asset = self.calc_asset()
 
            actions_index = actions.argsort()
            actions = actions.gather(1,actions_index)
            self.state[:,1:self.stock_dim + 1] = self.state[:,1:self.stock_dim + 1].gather(1,actions_index)
            self.state[:,self.stock_dim + 1:] = self.state[:,self.stock_dim + 1:].gather(1,actions_index)

            for i in range(actions.shape[1]):
                trade_shares = self.trade(i, actions[:,i])

            self.actions_memory.append(actions)

            ver_index = actions_index.argsort()
            actions = actions.gather(1,ver_index)
            self.state[:,1:self.stock_dim + 1] = self.state[:,1:self.stock_dim + 1].gather(1,ver_index)#股票价格归序
            self.state[:,self.stock_dim + 1:] = self.state[:,self.stock_dim + 1:].gather(1,ver_index)#shares归序
            
            self.day  += 1
            next_price = self.pred_close_price(self.history_price_seq, actions)
            self.state[:,1:self.stock_dim+1] = torch.repeat_interleave(next_price.reshape(1,30), self.state.shape[0], 0)
            # 当天进行交易之后的账户总资产
            end_total_asset = self.calc_asset()
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self.date)
            self.reward = (
                end_total_asset - begin_total_asset
            )  # reward = 今日账户变化（当天交易结束的账户总资产-当天起始时刻的账户总资产)
            self.rewards_memory.append(self.reward)
            self.reward = (
                self.reward * self.reward_scaling
            )  # self.reward_scaling = 0.0001
        return self.reward, self.terminal

    def init(self, state=None):
        self.state = state
        self.cost = 0
        self.trades_times = 0
        self.day = 0
        self.terminal = False

        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self.date]

    def reset(self, initial_state, history_price_seq):
        # initiate state
        self.history_price_seq = history_price_seq
        self.init(initial_state)

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if self.stock_dim > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.history_price_seq[-1]
                    + [0] * self.stock_dim
                )
            else:
                # for single stock
                # TODO
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        else:
            # Using Previous State
            if self.stock_dim > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.history_price_seq[-1]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                )
            else:
                # for single stock
                # TODO
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        return state

    def pred_close_price(self, history_price_seq, actions):
        # TODO Teather Forcing
        """[predict next price]

        Args:
            history_price_seq ([list]): [30,n_days]
            actions ([list]): [30]

        Returns:
            [list]: [30]
        """
        #暂时未用到actions，如需更改，则改动dynamic_model中的inference
        next_price = self.true_price_seq[:, self.day] if self.use_real_price else self.model_predict.inference(history_price_seq)
        return next_price

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if self.stock_dim > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def calc_asset(self):
        total_asset = self.state[:, 0] + ( \
                self.state[:, 1 : (self.stock_dim + 1)] \
                * self.state[:, (self.stock_dim + 1) : (self.stock_dim * 2 + 1)] \
            ).sum(dim=1)
        return total_asset
    
    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: Dict,
        num_particles: int,# 对每个sample出的action重复num_particles次分别计算reward再求平均，作为最终次action的rewards
    ) -> torch.Tensor:
        assert (
            len(action_sequences.shape) == 3
        )  # population_size(选出的action的数量), horizon（rollout步数),action_shape
        population_size, horizon, action_dim = action_sequences.shape
        history_price_seq = initial_state["history_price_seq"]
        initial_state = initial_state["initial_state"]

        initial_obs_batch = torch.repeat_interleave(
            initial_state.unsqueeze(0), num_particles * population_size, 0
        ).to(torch.float)

        self.reset(initial_obs_batch, history_price_seq)  # 重新设置env状态
        initial_asset = self.calc_asset()

        for time_step in range(horizon):
            actions_for_step = action_sequences[:, time_step, :]
            action_batch = torch.repeat_interleave(
                actions_for_step, num_particles, dim=0
            )
            _, _ = self.step(action_batch)

        #用rollout(time_step = k) k步 之后的asset - rollout(time_step = 0)进行rollout之前的资产值，表示total_reward
        total_rewards = self.calc_asset() - initial_asset

        total_rewards = total_rewards.reshape(-1, num_particles)
        return total_rewards.mean(dim=1)
