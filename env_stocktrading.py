from pathlib import Path
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import logger

import rsrs
import utils
import logging
import fin_stats
import hashlib


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        cfg,
        df,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        baseline_df,
        cache_dir: Path,
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        use_rsrs=True,
        logger: logging.Logger = None,
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.cache_dir = cache_dir

        self.logger = logger
        self.cfg = cfg

        # rsrs index
        self.ticker_list = self.df["tic"].unique().tolist()
        self.ticker_list.sort()
        self.dates = self.df["date"].unique().tolist()
        self.dates.sort()
        self.rsrs_start_date = cfg.market.rs_start_date
        self.rs_sell = cfg.market.rs_sell
        self.rs_buy = cfg.market.rs_buy
        self.rsrs_end_date = self.dates[0]

        self.ans = []
        self.ans_rightdev = []

        self.use_rsrs = use_rsrs
        if self.use_rsrs:
            self.init_rsrs()
        # initalize state
        self.state = self._initiate_state()
        self.baseline_df = baseline_df
        self.baseline_df["daliy_return"] = baseline_df["close"].pct_change().tolist()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.df_account_value_stat = None

        # stats over all episodes
        self.stats = [
            [],
            [],
            [],
            [],
        ]  # total steps, annual return, sharpe ratio, max down
        # self.reset()
        self._seed()

    def init_rsrs(self):
        print(f"init rsrs ...")
        tics = self.df.tic.unique().tolist()
        tics.sort()
        encoder = hashlib.sha256()
        encoder.update("_".join(list(tics)).encode())
        cache_path = self.cache_dir/ f"rsrs_{self.dates[0]}_{self.dates[-1]}_{self.cfg.market.rs_N}_{self.cfg.market.rs_M}_{encoder.hexdigest()}.npy"

        if cache_path.is_file():
            self.rsrs_rightdev = np.load(cache_path).tolist()
            self.logger.info(f"load rsrs from {cache_path}")
        else:
            # initiate rsrs
            self.ans, self.ans_rightdev = rsrs.init_rsrs(
                self.cfg.market.id,
                self.ticker_list,
                self.rsrs_start_date,
                self.rsrs_end_date,
                self.cfg.market.rs_N,
            )

            # 加入rsrs指数
            self.rsrs_rightdev = []
            for date in self.dates:
                self.rsrs_rightdev.append(
                    rsrs.update(
                        date,
                        self.ans,
                        self.ans_rightdev,
                        self.ticker_list,
                        self.cfg.market.id,
                        self.cfg.market.rs_M,
                        self.cfg.market.rs_N,
                    )
                )

            np_rsrs_rightdev = np.array(self.rsrs_rightdev)
            np.save(cache_path, np_rsrs_rightdev)

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + 1] > 0:
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct)
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct)
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * self.state[index + self.stock_dim + 1]
                            * self.sell_cost_pct
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.state[index + 1] > 0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = (
                    self.state[0] // self.state[index + 1]
                )  # //整除，得到最大可以买入的数量
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct)
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig("results/account_value_trade_{}.png".format(self.episode))
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.initial_amount
            )
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
            # if self.episode % self.print_verbosity == 0:
            # account value df
            df_account_value_stat = pd.DataFrame(
                {"account_value": self.asset_memory, "date": self.dates}
            )
            df_account_value_stat["daily_return"] = (
                df_account_value_stat["account_value"].pct_change().tolist()
            )
            df_account_value_stat["datadate"] = self.dates
            self.df_account_value_stat = df_account_value_stat

            stat_info = fin_stats.stat_all(df_account_value_stat)
            if self.logger is not None:

                self.logger.info(
                    f"\nduration: from {self.dates[0]} to {self.dates[-1]}, days: {self.day}, episodes: {self.episode}\n"
                    f"begin_total_asset: {self.asset_memory[0]:0.2f}\n"
                    f"end_total_asset: {end_total_asset:0.2f}\n"
                    f"total_reward: {tot_reward:0.2f}\n"
                    f"total_cost: {self.cost:0.2f}\n"
                    f"total_trades: {self.trades}\n"
                    f"{stat_info}"
                )
            stat_info = pd.DataFrame(stat_info)
            aunual_return, sharpe, max_drawdown = (
                stat_info[0][0],
                stat_info[0][3],
                stat_info[0][6],
            )
            self.stats[0].append(self.episode * len(self.dates) + self.day)
            self.stats[1].append(aunual_return)
            self.stats[2].append(sharpe)
            self.stats[3].append(max_drawdown)

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
            # logger.record("environment/total_trades", self.trades)

            # return self.state, self.reward, self.terminal
            return self.state, self.reward, self.terminal, {}

        else:
            actions = np.ceil(
                actions * self.hmax
            )  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            # mod:int = 10
            # actions 的绝对值为mod的整数倍
            # actions = (actions//mod)*mod

            # 对30只股票分别判断rsrs指标
            if self.use_rsrs:
                for i in range(self.stock_dim):
                    if self.rsrs_rightdev[self.day][i] < self.rs_sell:
                        actions[i] = -1 * self.hmax
                    if self.rsrs_rightdev[self.day][i] > self.rs_buy:
                        actions[i] = self.hmax

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset)),begin_total_asset，当天起始时刻的账户总资产

            argsort_actions = np.argsort(actions)
            # 找到临界点，取0以下的index为sell stock index，0以上index为buy stock index
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                self.turbulence = self.data["turbulence"].values[0]
            self.state = self._update_state()
            # 当天进行交易之后的账户总资产
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())

            # reward 1: 资产增加值
            # self.reward = (
            #    end_total_asset - begin_total_asset
            # )  # reward = 今日账户变化（当天交易结束的账户总资产-当天起始时刻的账户总资产)
            # reward 2： 资产增加百分比
            self.reward = (
                100 * (end_total_asset - begin_total_asset) / begin_total_asset
            )
            # self.reward = 100*((end_total_asset - begin_total_asset) / begin_total_asset-self.baseline_df.iloc[self.day]["daliy_return"])

            self.rewards_memory.append(self.reward)
            self.reward = (
                self.reward * self.reward_scaling
            )  # self.reward_scaling = 0.0001

        # return self.state, self.reward, self.terminal
        return self.state, self.reward, self.terminal, {}

    def reset(self, day=0):
        # initiate state
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.day = day
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.df_account_value_stat = None

        self.episode += 1

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + [0] * self.stock_dim
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                    # + self.rsrs_rightdev[0]
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock；state for last moment; data for current moment
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    [
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ],
                    [],
                )
                # + self.rsrs_rightdev[self.day]
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                # +self.rsrs_rightdev[self.day]
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

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
        if len(self.df.tic.unique()) > 1:
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

    def save_stats(self):
        df_stats = pd.DataFrame(
            {
                "steps": self.stats[0],
                "Annual return": self.stats[1],
                "Sharpe ratio": self.stats[2],
                "Max drawdown": self.stats[3],
            }
        )
        return df_stats

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def calc_asset(self):
        total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        )
        return total_asset
