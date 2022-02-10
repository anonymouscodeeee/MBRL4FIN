import numpy as np
from datetime import datetime, timedelta  # 记录outputs_time，记录循环用时
from typing import Optional, Tuple, cast

import hydra.utils
import numpy as np
import omegaconf
from omegaconf import dictconfig
import torch
import math

# matplotlib.use('Agg')
#%matplotlib inline
import torch
import omegaconf
from pathlib import Path
import logging
import shutil
import sys, os
import pickle
import plot
import analysis

current_dir = Path(os.path.realpath("."))



from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer

from mbrl.models import (
    OneDTransitionRewardModel,
    ModelEnv,
    ModelTrainer,
)

import mbrl.types as mbrl_types
import mbrl.util as mbrl_util
import mbrl.util.common as mbrl_common
import mbrl.constants as mbrl_constants
import mbrl.planning as mbrl_planning
from mbrl.util import ReplayBuffer
import mbrl.third_party.pytorch_sac as pytorch_sac
from env_stocktrading import StockTradingEnv
import plot
from mbrl_fin import MBRLFin


MBPO_LOG_FORMAT = mbrl_constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]


class MBPOFin(MBRLFin):
    def __init__(
        self,
        cfg: omegaconf.dictconfig,
        flogger: logging.Logger,
        checkpoint_dir: Path,
        checkpoint: str = None,
    ):
        super().__init__(cfg, flogger, checkpoint_dir, checkpoint)

        self.dynamics_model: OneDTransitionRewardModel = None
        self.agent: pytorch_sac.SACAgent = None

        self.dynamics_model_stat  = []

    def create_models(self, env: StockTradingEnv):
        mbrl_planning.complete_agent_cfg(env, self.cfg.algorithm.agent)
        self.agent: pytorch_sac.SACAgent = hydra.utils.instantiate(
            self.cfg.algorithm.agent
        )
        self.dynamics_model: OneDTransitionRewardModel = (
            mbrl_common.create_one_dim_tr_model(
                self.cfg, env.observation_space.shape, env.action_space.shape
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
        # save sac
        self.agent.save(save_dir)

    def load_checkpoint(self):
        checkpoint = self.checkpoint_dir / self.checkpoint
        assert checkpoint.is_dir()
        config_file = "config.yaml"
        config_path = Path(f"./.hydra/{config_file}")
        shutil.copyfile(checkpoint / f"{config_file}", config_path)
        self.dynamics_model.load(checkpoint)
        self.agent.load(checkpoint)

    def rollout_model_and_populate_sac_buffer(
        self,
        model_env: ModelEnv,
        replay_buffer: ReplayBuffer,
        sac_buffer: pytorch_sac.ReplayBuffer,
        sac_samples_action: bool,
        rollout_horizon: int,
        batch_size: int,
    ):

        batch = replay_buffer.sample(batch_size)
        initial_obs, *_ = cast(mbrl_types.TransitionBatch, batch).astuple()
        obs = model_env.reset(
            initial_obs_batch=cast(np.ndarray, initial_obs),
            return_as_np=True,
        )
        accum_dones = np.zeros(obs.shape[0], dtype=bool)
        accum_masked = np.ones(obs.shape[0], dtype=bool)

        # dynamic model stats
        current_obs = []
        current_actions = []
        next_obs = []
        next_rewards = []
        next_dones = []
        for i in range(rollout_horizon):
            action = self.agent.act(obs, sample=sac_samples_action, batched=True)
            model_env.dynamics_model.model.update_mask_ratio(rollout_horizon, i)
            current_obs.append(model_env._current_obs.clone().detach().cpu())
            current_actions.append(action)
            pred_next_obs, pred_rewards, pred_dones, _ = model_env.step(
                action, sample=True
            )
            next_obs.append(pred_next_obs)
            next_rewards.append(pred_rewards)
            next_dones.append(pred_dones)

            filters = ~accum_dones
            # only for "mask"
            if self.cfg.dynamics_model.model.propagation_method == "mask":
                mask = model_env.dynamics_model.model.mask.cpu().detach().numpy()
                uncertainty = (
                    model_env.dynamics_model.model.uncertainty.cpu()
                    .detach()
                    .unsqueeze(1)
                    .numpy()
                )
                pred_rewards -= uncertainty * self.cfg.dynamics_model.mask_penalty
                accum_masked &= mask
                if self.cfg.dynamics_model.mask_mode == "nonstop":
                    filters &= mask
                elif self.cfg.dynamics_model.mask_mode == "hardstop":
                    filters &= accum_masked
                else:
                    assert "Unknown mask mode."

            pred_dones = pred_dones.reshape((pred_dones.shape[0], 1))
            sac_buffer.add_batch(
                obs[filters],
                action[filters],
                pred_rewards[filters],
                pred_next_obs[filters],
                pred_dones[filters],
                pred_dones[filters],
            )
            obs = pred_next_obs
            # must not done
            accum_dones &= ~pred_dones.squeeze()

        self.dynamics_model_stat.append(
            {
                "initial_batch":batch,
                "obs":np.concatenate(current_obs,axis=0),
                "actions":np.concatenate(current_actions, axis=0),
                "next_obs":np.concatenate(next_obs, axis=0),
                "next_rewards":np.concatenate(next_rewards, axis=0),
                "next_dones":np.concatenate(next_dones, axis=0),
            }
        )

    def maybe_replace_sac_buffer(
        self,
        sac_buffer: Optional[pytorch_sac.ReplayBuffer],
        new_capacity: int,
        obs_shape: Tuple[int],
        act_shape: Tuple[int],
    ):
        if sac_buffer is None or new_capacity != sac_buffer.capacity:
            new_buffer = pytorch_sac.ReplayBuffer(
                obs_shape, act_shape, new_capacity, self.cfg.device
            )
            if sac_buffer is None:
                return new_buffer
            n = len(sac_buffer)
            new_buffer.add_batch(
                sac_buffer.obses[:n],
                sac_buffer.actions[:n],
                sac_buffer.rewards[:n],
                sac_buffer.next_obses[:n],
                np.logical_not(sac_buffer.not_dones[:n]),
                np.logical_not(sac_buffer.not_dones_no_max[:n]),
            )
            return new_buffer
        return sac_buffer

    def termination_fn(self, actions: torch.tensor, next_observs: torch.tensor):
        return torch.full((actions.shape[0], 1), False, device=actions.device)

    def save_stats(self):
        with open("mbpo_stats.pkl", "wb") as f:
            pickle.dump(self.dynamics_model_stat, f)

    def train(
        self,
        train_env: StockTradingEnv,
        val_env: StockTradingEnv,
    ):
        obs_shape = train_env.observation_space.shape
        act_shape = train_env.action_space.shape

        self.agent.stat_path = "Q_loss.npy"

        # if os.path.exists(critic_path):
        #    agent.critic.load_state_dict(torch.load(critic_path))
        # if os.path.exists(actor_path):
        #    agent.actor.load_state_dict(torch.load(actor_path))
        # """

        work_dir = os.getcwd()
        # enable_back_compatible to use pytorch_sac agent
        logger = mbrl_util.Logger(work_dir, enable_back_compatible=True)
        logger.register_group(
            mbrl_constants.RESULTS_LOG_NAME,
            MBPO_LOG_FORMAT,
            color="green",
            dump_frequency=1,
        )

        rng = np.random.default_rng(seed=self.cfg.seed)
        torch_generator = torch.Generator(device=self.cfg.device)
        if self.cfg.seed is not None:
            torch_generator.manual_seed(self.cfg.seed)

        # -------------- Create initial overrides. dataset --------------
        use_double_dtype = self.cfg.algorithm.get("normalize_double_precision", False)
        dtype = np.double if use_double_dtype else np.float32
        replay_buffer: ReplayBuffer = mbrl_common.create_replay_buffer(
            self.cfg,
            obs_shape,
            act_shape,
            rng=rng,
            obs_type=dtype,
            action_type=dtype,
            reward_type=dtype,
        )
        random_explore = self.cfg.algorithm.random_initial_explore
        self.flogger.info("rollout_agent_trajectories ...")
        mbrl_common.rollout_agent_trajectories(
            train_env,
            self.cfg.algorithm.initial_exploration_steps,  # 从真实环境中采样的长度，可能要使用整个train data的天数
            mbrl_planning.RandomAgent(train_env) if random_explore else self.agent,
            {} if random_explore else {"sample": True, "batched": False},
            replay_buffer=replay_buffer,
        )

        epoch_length = self.cfg.overrides.epoch_length
        # epoch_length = len(env.dates)

        # ---------------------------------------------------------
        # --------------------- Training Loop ---------------------
        rollout_batch_size = (
            self.cfg.overrides.effective_model_rollouts_per_step
            * self.cfg.algorithm.freq_train_model
        )
        trains_per_epoch = int(
            np.ceil(epoch_length / self.cfg.overrides.freq_train_model)
        )
        updates_made = 0
        env_steps = 0
        model_env = ModelEnv(
            train_env,
            self.dynamics_model,
            self.termination_fn,
            None,
            generator=torch_generator,
        )
        model_trainer = ModelTrainer(
            self.dynamics_model,
            optim_lr=self.cfg.overrides.model_lr,
            weight_decay=self.cfg.overrides.model_wd,
            logger=logger,
        )
        metric_best = (-np.inf, -np.inf, -np.inf)
        epoch = 0
        sac_buffer = None
        early_stop = False
        eval_times = 0
        metric_buff = []
        while env_steps < self.cfg.overrides.num_steps and not early_stop:
            # 此处决定了rollout的步长，现有逻辑恒定为epoch + 1，因为cfg.overrides.rollout_schedule=[1,15,1,1],可能要进行修改
            # 具体数学逻辑参考/mnt/guiyi/hhf/mbrl/mbrl/util/math.py:16 truncated_linear

            if self.cfg.overrides.dynamic_rollout:
                rollout_length = int(
                    mbrl_util.truncated_linear(
                        1,
                        math.ceil(self.cfg.overrides.num_steps / epoch_length),
                        self.cfg.overrides.rollout_schedule[0],
                        self.cfg.overrides.rollout_schedule[1],
                        epoch + 1,
                    )
                )
            else:
                rollout_length = 1

            if self.cfg.dynamics_model.model.propagation_method == "mask":
                sac_buffer_capacity = (
                    rollout_length
                    * max(
                        1,
                        int(
                            rollout_batch_size
                            * self.cfg.dynamics_model.model.min_mask_ratio
                        ),
                    )
                    * trains_per_epoch
                )
            else:
                sac_buffer_capacity = (
                    rollout_length * rollout_batch_size * trains_per_epoch
                )
            sac_buffer_capacity *= self.cfg.overrides.num_epochs_to_retain_sac_buffer
            sac_buffer = self.maybe_replace_sac_buffer(
                sac_buffer,
                sac_buffer_capacity,
                obs_shape,
                act_shape,
            )
            obs, done = None, False
            for steps_epoch in range(epoch_length):
                if (
                    steps_epoch == 0 or done
                ):  # 则最多只会利用env中的epoch_length天的数据进行step，可能要改为train data的全部天数
                    self.flogger.info("reset train env")
                    obs, done = train_env.reset(), False
                # --- Doing env step and adding to model dataset ---
                next_obs, reward, done, _ = mbrl_common.step_env_and_add_to_buffer(
                    train_env, obs, self.agent, {}, replay_buffer
                )

                # --------------- Model Training -----------------
                if (
                    env_steps + 1
                ) % self.cfg.overrides.freq_train_model == 0:  # 环境模型的训练频率，根据总的step数目来确定
                    self.flogger.info("training dynamic model ...")
                    mbrl_common.train_model_and_save_model_and_data(
                        self.dynamics_model,
                        model_trainer,
                        self.cfg.overrides,
                        replay_buffer,
                        work_dir=work_dir,
                    )
                    # --------- Rollout new model and store imagined trajectories --------
                    # Batch all rollouts for the next freq_train_model steps together
                    self.flogger.info(f"env_steps: {env_steps}, rollout ...")
                    self.rollout_model_and_populate_sac_buffer(
                        model_env,
                        replay_buffer,
                        sac_buffer,
                        self.cfg.algorithm.sac_samples_action,
                        rollout_length,  # rollout步长
                        rollout_batch_size,  # batch
                    )

                # --------------- Agent Training -----------------
                for _ in range(
                    self.cfg.overrides.num_sac_updates_per_step
                ):  # 每次step中，agent更新的次数
                    if (
                        env_steps + 1
                    ) % self.cfg.overrides.sac_updates_every_steps != 0 or len(
                        sac_buffer
                    ) < rollout_batch_size:
                        break  # only update every once in a while
                    self.agent.update(sac_buffer, logger, updates_made)
                    updates_made += 1
                    if updates_made % self.cfg.log_frequency_agent == 0:
                        logger.dump(updates_made, save=True)

                # ------ Epoch ended (evaluate and save model) ------
                if (env_steps + 1) % self.cfg.overrides.freq_evaluate == 0:  # 进行估值评价的频率
                    self.flogger.info(f"env_steps: {env_steps}, evaluating ...")

                    _, df_stat, pred_actions = self.evaluate(val_env)
                    #当年收益大于14%，sharpe>1,max_drawdown>-0.1,且保持此状态超过4次evaluae
                    target_metric = list(self.cfg.overrides.target_metric)#(0.16, 1.2, -0.11)

                    # aunual_return, sharpe, max_drawdown = stats[0], stats[3],stats[6]
                    metric = [df_stat[0], df_stat[3], df_stat[6]]
                    #存放最近的几次metric结果
                    
                    buff_size = 3
                    if len(metric_buff) <buff_size:
                        metric_buff.append(metric)
                    else:
                        metric_buff[eval_times%buff_size]=metric
                    eval_times += 1
                    self.flogger.info(f"metric_buff: {metric_buff}, targe: {target_metric}")
                    if len(metric_buff)==buff_size: 
                        all_match = True
                        for i in range(len(metric_buff)):
                            if metric_buff[i]<target_metric:
                                all_match = False
                                break
                        self.print_movement_stats(val_env, pred_actions)
                        if (
                            env_steps > self.cfg.overrides.save_min_steps
                            and all_match
                        ):  
                            self.flogger.info(f"do early stop, steps: {env_steps} .")
                            metric_best = metric
                            self.save_checkpiont()
                            early_stop = True and self.cfg.overrides.use_earlystop
                            break

                env_steps += 1
                obs = next_obs
            epoch += 1


        self.flogger.info("train done.")
        self.agent.save_stats()
        plot.draw_loss(self.agent.stat_path)
        #这个统计结果的数据很大，可能十几个G以上，默认不会存下来。
        if self.cfg.overrides.save_rollout_stats:
            self.save_stats()
        analysis.draw(self.dynamics_model_stat, Path("./"))
        return np.float32(metric_best)
