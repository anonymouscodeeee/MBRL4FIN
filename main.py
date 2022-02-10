# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import hydra
import numpy as np
import omegaconf
import torch
import utils
import os
from pathlib import Path
import math


from datetime import datetime, timedelta
from hydra._internal.config_search_path_impl import (
    ConfigSearchPathImpl as hydra_ConfigSearchPathImpl,
)

from hydra._internal.config_repository import ConfigRepository as hydra_ConfigRepository

import data_process
import plot
import mbpo_fin, pets_fin
from env_stocktrading import StockTradingEnv
import mbrl_fin

ROOT_DIR = Path(os.path.realpath("."))
DATASETS_DIR = ROOT_DIR / "datasets"
CACHE_DIR = ROOT_DIR / "outputs/cache"
CHECK_POINT_DIR = ROOT_DIR / "checkpoint"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECK_POINT_DIR.mkdir(parents=True, exist_ok=True)


def restore_cfg(path: Path, cfg):
    config_search_path = hydra_ConfigSearchPathImpl()
    config_search_path.append(path.name[:-5], str(path.parent))
    config_repo = hydra_ConfigRepository(config_search_path=config_search_path)
    ret = config_repo.load_config(
        config_path=path.name[:-5],
        is_primary_config=False,
    )
    # cfg.restore 中的内容保持不变
    ret.config.restore = cfg.restore
    cfg = ret.config


def data_prepare(cfg):
    # 如果有cache不会再重新生成取数据
    tick_list = data_process.tick_pick(
        DATASETS_DIR,
        cfg.market.id,
        list(cfg.market.tickers),
        ascending=cfg.bottom_select,
    )
    # tick_list = data_process.tick_pick(
    #    DATASETS_DIR, cfg.market.id, list(cfg.market.tickers))
    assert len(tick_list) == cfg.stock_num

    _, df_train, df_val, df_test = data_process.preprocess(
        DATASETS_DIR,
        cfg.market.id,
        cfg.dates.start_date,
        cfg.dates.end_date,
        tick_list,
        cfg.dates.train_start,
        cfg.dates.train_end,
        cfg.dates.val_start,
        cfg.dates.val_end,
        cfg.dates.test_start,
        cfg.dates.test_end,
        cfg.market.tech_indicators,
        CACHE_DIR,
    )

    return df_train, df_val, df_test


def create_env(cfg, df_train, df_val, df_test, flogger):
    stock_dimension = cfg.stock_num
    state_space = (
        1 + 2 * stock_dimension + len(cfg.market.tech_indicators) * stock_dimension
    )
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    print(cfg.market.tech_indicators)
    env_kwargs_common = {
        "cfg": cfg,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": cfg.market.tech_indicators,
        "action_space": stock_dimension,
        "logger": flogger,
        "cache_dir": CACHE_DIR,
    }

    def _create_one(market_id, df, start_date, end_date, seed):
        baseline_df = plot.get_cached_baseline(market_id, start_date, end_date)
        env = StockTradingEnv(
            df=df,
            baseline_df=baseline_df,
            **env_kwargs_common,
            **dict(cfg.env),
        )
        env.seed(seed)
        return env

    env_train = _create_one(
        cfg.market.id, df_train, cfg.dates.train_start, cfg.dates.train_end, 0
    )
    env_val = _create_one(
        cfg.market.id, df_val, cfg.dates.val_start, cfg.dates.val_end, 0
    )
    env_test = _create_one(
        cfg.market.id, df_test, cfg.dates.test_start, cfg.dates.test_end, 1
    )

    return env_train, env_val, env_test


MODELS = {
    "mbpo": mbpo_fin.MBPOFin,
    "pets": pets_fin.PETSFin,
}


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    checkpoint = None
    if cfg.restore.checkpoint != "none":
        restore_cfg(CHECK_POINT_DIR / f"{cfg.restore.checkpoint}/config.yaml", cfg)
        checkpoint = CHECK_POINT_DIR / cfg.restore.checkpoint
    flogger = utils.setup_logger(cfg.market.name, f"info.log")
    flogger.info(f"config has been successfully loaded, pid {os.getpid()} .")
    df_train, df_val, df_test = data_prepare(cfg)
    env_train, env_val, env_test = create_env(cfg, df_train, df_val, df_test, flogger)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model: mbrl_fin.MBRLFin = MODELS[cfg.algorithm.name](
        cfg,
        flogger,
        CHECK_POINT_DIR,
        checkpoint,
    )
    model.create_models(env_test)  # train 和test均可以，本质上是将obs和action信息传进去（shape)

    if checkpoint is not None:
        model.load_checkpoint()

    if not cfg.restore.test_only:
        time_train_start = datetime.now()
        model.train(env_train, env_val)
        train_duration = math.ceil(
            (datetime.now() - time_train_start).total_seconds() / 60
        )
        flogger.info(f"train duration : {train_duration} minutes .")

    flogger.info("====================evaluating====================")
    model.evaluate(env_test, True)

    train_stats_path = f"train_stats.csv"
    env_train.save_stats().to_csv(train_stats_path)
    flogger.info(f"train stats path : {str(train_stats_path)}")

    val_stats_path = f"val_stats.csv"
    env_val.save_stats().to_csv(val_stats_path)
    flogger.info(f"val stats path : {str(val_stats_path)}")

    test_stats_path = f"test_stats.csv"
    env_test.save_stats().to_csv(test_stats_path)
    flogger.info(f"test stats path : {str(test_stats_path)}")


if __name__ == "__main__":
    run()
