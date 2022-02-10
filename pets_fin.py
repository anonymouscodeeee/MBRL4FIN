import pandas as pd
import numpy as np
from datetime import datetime, timedelta  # 记录outputs_time，记录循环用时
import numpy as np
import omegaconf
import torch
import logging
import os, sys
import torch
from pathlib import Path
import shutil


from mbrl.models import (
    OneDTransitionRewardModel,
    ModelEnv,
    ModelTrainer,
)
import mbrl.constants as mbrl_constants
import mbrl.util as mbrl_util
import mbrl.util.common as mbrl_common
import mbrl.planning as mbrl_planning
from mbrl.util import ReplayBuffer

from env_stocktrading import StockTradingEnv
from mbrl_fin import MBRLFin

EVAL_LOG_FORMAT = mbrl_constants.EVAL_LOG_FORMAT


class PETSFin(MBRLFin):
    def __init__(
        self,
        cfg: omegaconf.dictconfig,
        flogger: logging.Logger,
        checkpoint_dir: Path,
        checkpoint: str = None,
    ):
        super().__init__(cfg, flogger, checkpoint_dir, checkpoint)

        self.dynamics_model: OneDTransitionRewardModel = None
        self.agent: mbrl_planning.TrajectoryOptimizerAgent = None

    def create_models(self, env: StockTradingEnv):
        self.dynamics_model = mbrl_util.common.create_one_dim_tr_model(
            self.cfg, env.observation_space.shape, env.action_space.shape
        )

        torch_generator = torch.Generator(device=self.cfg.device)
        if self.cfg.seed is not None:
            torch_generator.manual_seed(self.cfg.seed)
        self.model_env = ModelEnv(
            env,
            self.dynamics_model,
            self.termination_fn,
            self.reward_fn,
            generator=torch_generator,
        )
        self.agent: mbrl_planning.TrajectoryOptimizerAgent = (
            mbrl_planning.create_trajectory_optim_agent_for_model(
                self.model_env,
                self.cfg.algorithm.agent,
                num_particles=self.cfg.algorithm.num_particles,
            )
        )

    def save_checkpiont(self):
        time_current = datetime.now() + timedelta(hours=8)
        time_current = time_current.strftime("%Y-%m-%d_%H:%M:%f")
        save_dir = (
            self.checkpoint_dir
            / f"{self.cfg.algorithm.name}_{self.cfg.market.name}_{time_current}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        # save configs
        config_file = "config.yaml"
        config_path = Path(f"./.hydra/{config_file}")
        shutil.copyfile(config_path, save_dir / f"{config_file}")
        # save env model
        self.dynamics_model.save(save_dir)

    def load_checkpoint(self):
        checkpoint = self.checkpoint_dir / self.checkpoint
        assert checkpoint.is_dir()
        config_file = "config.yaml"
        config_path = Path(f"./.hydra/{config_file}")
        shutil.copyfile(checkpoint / f"{config_file}", config_path)
        self.dynamics_model.load(checkpoint)

    def termination_fn(self, actions: torch.tensor, next_observs: torch.tensor):
        return torch.full((actions.shape[0], 1), False, device=actions.device)

    def reward_fn(self, actions: torch.tensor, next_observs: torch.tensor):
        stock_num = actions.shape[-1]
        total_asset = next_observs[:, 0:1] + (
            next_observs[:, 1 : (stock_num + 1)]
            * next_observs[:, (stock_num + 1) : (stock_num * 2 + 1)]
        ).sum(dim=1, keepdim=True)
        return total_asset

    def train(
        self,
        train_env: StockTradingEnv,
        test_env: StockTradingEnv,
    ):
        # ------------------- Initialization -------------------
        rng = np.random.default_rng(seed=self.cfg.seed)

        work_dir = os.getcwd()
        print(f"Results will be saved at {work_dir}.")

        logger = mbrl_util.Logger(work_dir)
        logger.register_group(
            mbrl_constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )

        # -------- Create and populate initial env dataset --------
        use_double_dtype = self.cfg.algorithm.get("normalize_double_precision", False)
        dtype = np.double if use_double_dtype else np.float32
        replay_buffer = mbrl_common.create_replay_buffer(
            self.cfg,
            train_env.observation_space.shape,
            train_env.action_space.shape,
            rng=rng,
            obs_type=dtype,
            action_type=dtype,
            reward_type=dtype,
        )
        mbrl_common.rollout_agent_trajectories(
            train_env,
            self.cfg.algorithm.initial_exploration_steps,
            mbrl_planning.RandomAgent(train_env),
            {},
            replay_buffer=replay_buffer,
        )
        replay_buffer.save(work_dir)

        # ---------------------------------------------------------
        # ---------- Create model environment and agent -----------
        model_trainer = ModelTrainer(
            self.dynamics_model,
            optim_lr=self.cfg.overrides.model_lr,
            weight_decay=self.cfg.overrides.model_wd,
            logger=logger,
        )

        # ---------------------------------------------------------
        # --------------------- Training Loop ---------------------
        env_steps = 0
        current_trial = 0
        metric_best = (-np.inf, -np.inf, -np.inf)
        while env_steps < self.cfg.overrides.num_steps:
            obs = train_env.reset()
            self.agent.reset()
            done = False
            total_reward = 0.0
            steps_trial = 0

            while not done:
                # --------------- Model Training -----------------
                if env_steps % self.cfg.algorithm.freq_train_model == 0:
                    mbrl_common.train_model_and_save_model_and_data(
                        self.dynamics_model,
                        model_trainer,
                        self.cfg.overrides,
                        replay_buffer,
                        work_dir=work_dir,
                    )

                # --- Doing env step using the agent and adding to model dataset ---
                next_obs, reward, done, _ = mbrl_util.common.step_env_and_add_to_buffer(
                    train_env, obs, self.agent, {}, replay_buffer
                )

                obs = next_obs
                total_reward += reward
                steps_trial += 1
                env_steps += 1

            self.flogger.info("evaluating.......")
            _, df_stat, pred_actions = self.evaluate(test_env)
            # aunual_return, sharpe, max_drawdown = stats[0][0], stats[0][3],stats[0][6]
            metric = (df_stat[0], df_stat[3], df_stat[6])
            if env_steps > self.cfg.overrides.save_min_steps and metric > metric_best:
                metric_best = metric
                self.save_checkpiont()
                pass
            # 尚未保存agent
            current_trial += 1
