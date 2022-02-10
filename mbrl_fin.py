from env_stocktrading import StockTradingEnv
import torch
import logging
from datetime import datetime, timedelta
import omegaconf
import fin_stats
import numpy as np
import plot
from pathlib import Path
from sklearn.metrics import accuracy_score


class MBRLFin:
    def __init__(
        self,
        cfg: omegaconf.dictconfig,
        flogger: logging.Logger,
        checkpoint_dir: Path,
        checkpoint: Path = None,
    ):
        self.cfg = cfg
        self.flogger = flogger
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = checkpoint
        self.agent = None

    def create_models(self, env:StockTradingEnv):
        pass

    @torch.no_grad()
    def evaluate(
        self,
        env: StockTradingEnv,
        show_baseline=False,
    ):
        assert self.agent is not None

        time_current = datetime.now() + timedelta(hours=8)
        time_current = time_current.strftime("%Y-%m-%d_%H:%M")
        csv_name_func = lambda time: f"{time}_account_value.csv"
        date_val = env.df["date"].unique().tolist()
        date_val.sort()
        pred_actions = []
        obs = env.reset()

        # video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        while not done:
            action = self.agent.act(obs)
            pred_actions.append(action)
            obs, reward, done, _ = env.step(action)

            # print(f"episode {episode}: account_value===={account_value[-1]}====")
            episode_reward += reward

        pred_actions = np.array(pred_actions)
        self.flogger.info(f"save evaluation to {csv_name_func(time_current)} .")
        env.df_account_value_stat.to_csv(csv_name_func(time_current))

        df_stat = fin_stats.stat_all(env.df_account_value_stat)

        if show_baseline:
            self.flogger.info("\nbase line stats:")
            plot.show_baseline(self.cfg.market.id, env.dates[0], env.dates[-1])
        # aunual_return, sharpe, max_drawdown = stats[0][0][0], stats[0][0][3],stats[0][0][6]

        return (
            env.df_account_value_stat,
            df_stat,
            pred_actions,
        )  # df_stat[0][0], df_stat[0][3], df_stat[0][6]

    def print_movement_stats(self, env: StockTradingEnv, pred_actions: np.array):
            # [tic, day]
        close_prices = torch.from_numpy(
            env.df.sort_values(["tic", "date"]).close.to_numpy()
        ).view(-1, len(env.df.date.unique()))
        # [tic, day]
        pred_actions = torch.from_numpy(pred_actions).permute(1, 0)
        prices_change = close_prices[:, 1:] - close_prices[:, :-1]
        movements = (prices_change > 0).flatten()
        pred_movements = (pred_actions[:, :-1] > 0).flatten()

        potential_reward = prices_change.abs().sum()
        action_reward = (pred_actions[:, :-1] * prices_change).sum()
        reward_score = (action_reward / potential_reward).item()
        accuracy = accuracy_score(movements, pred_movements)
        print(
            f"movement prediction accuracy:{accuracy:0.4f}, reward score:{reward_score:0.4f}"
        )

    def train(
        self,
        train_env: StockTradingEnv,
        test_env: StockTradingEnv,
    ):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError
