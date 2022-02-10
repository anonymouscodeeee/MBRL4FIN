import pandas as pd
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import empyrical as ep
from pathlib import Path
from scipy import optimize

import plot
import fin_stats

from finrl.plot import get_daily_return

OUTDIR = Path(os.path.realpath(".") + "/paper")

PATHS = [
    Path("outputs/2022-02-09/mbpo_sp/rsac_8000/train.csv"),  # RSAC
    Path("outputs/2022-02-09/mbpo_sp/rspo_8000/train.csv"),  # RSPO
    Path("outputs/2022-02-05/mbpo_sp/m2ac_2/train.csv"),  # M2AC
    Path("outputs/2022-02-09/mbpo_sp/mbpo_8000/train.csv"),  # MBPO
    Path("outputs/results/bottom/pets/2021-12-06_00:10_account_value.csv"),  # PETS
]

labels = ["RSAC", "RSPO", "M2AC", "MBPO", "PETS", "Baseline"]
colors = ["#ff7f00", "#7590b9", "#beaed4", "#fb9a99", "#80b1d3", "#A6A6A6"]
lines = ["-", "-", "-", "-", "-", "-", "-"]


base_df = plot.get_cached_baseline(2, "2018-07-01", "2021-07-01")
base_cu = ep.cum_returns(get_daily_return(base_df, "close"), 1000000)

dfs = [pd.read_csv(path).account_value for path in PATHS]
dfs.append(base_cu)


def split_results(df_total):
    durations = [
        ["2018-07-01", "2019-07-01"],
        ["2019-07-01", "2020-07-01"],
        ["2020-07-01", "2021-07-01"],
    ]

    for i, duration in enumerate(durations):
        print(f"episode {i}:")
        df = df_total[(df_total.date >= duration[0]) & (df_total.date < duration[1])]
        stats = fin_stats.stat_all(df)
        print(stats)


df_1 = pd.read_csv(PATHS[0])
stats = fin_stats.stat_all(df_1)
print(stats)
# split_results(df_1)


def draw_returns(df_all):
    # markers=["o","s","^","*"]
    x = range(len(df_all[0]))  # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plt.figure(0)
    for i, df in enumerate(df_all):
        plt.plot(x, df, color=colors[i], linestyle=lines[i], label=labels[i])
    plt.xlabel("Day", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.ylabel("Account value", fontsize=14)
    plt.title("")
    plt.legend((labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]))
    plt.savefig(OUTDIR / "returns.pdf", format="pdf")



def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def draw_log_critic_loss():
    plt.figure(1)
    for i, path in enumerate(PATHS):
        loss_filepath = path.parent / "train.csv"
        if not loss_filepath.is_file():
            continue
        df_loss = pd.read_csv(path.parent / "train.csv")
        log_critic_loss = df_loss.critic_loss[10:81].apply(np.log)
        a, b, c = optimize.curve_fit(func, range(10, 81), log_critic_loss)[0]
        x = np.arange(10, 81, 1)
        y = a * np.exp(-b * x) + c
        plt.plot(x, y, color=colors[i], linestyle=lines[i], label=labels[i])
    plt.xlabel("Training episode", fontsize=14)
    plt.tick_params(labelsize=8)
    plt.ylabel("Log of critic loss", fontsize=14)
    plt.title("")
    plt.legend((labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]))

    for i, path in enumerate(PATHS):
        loss_filepath = path.parent / "train.csv"
        if not loss_filepath.is_file():
            continue
        df_loss = pd.read_csv(path.parent / "train.csv")
        log_critic_loss = df_loss.critic_loss[10:71].apply(np.log)

        plt.scatter(range(10, 81), log_critic_loss, s=20, color=colors[i], alpha=0.3)
    plt.savefig(OUTDIR / "log_critic_loss.pdf", format="pdf")
    plt.show()


draw_returns(dfs)
# draw_critic_loss()
draw_log_critic_loss()


# 画出各种算法的年收益曲线
df_1 = pd.read_csv(PATHS[1])
stats = fin_stats.stat_all(df_1)
print(stats)
