import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os, sys
from pathlib  import Path
import numpy as np

BASELINE_CACHE_PATHS = [
    "datasets/baseline_dow_2000-01-01_2021-11-01.csv",
    "datasets/baseline_nas_2000-01-01_2021-11-01.csv",
    "datasets/baseline_sp_2000-01-01_2021-11-01.csv",
]

ROOT_DIR = Path(os.path.realpath("."))

from finrl.plot import (
    backtest_stats,
    backtest_plot,
    get_daily_return,
    get_baseline,
)

def show_baseline(index, start_date, end_date):
    df=get_cached_baseline(index, start_date, end_date)
    backtest_stats(df, value_col_name="close")

def get_cached_baseline(index, start_date, end_date):
    df = pd.read_csv(ROOT_DIR/BASELINE_CACHE_PATHS[index])
    df = df[(df.date >= start_date) & (df.date < end_date)]
    return df


def account_value_show(df, market_id=0, name="account_value"):
    #baseline_tickers = ["^DJI", "^NDX", "^GSPC"]
    df[f"{name}"].plot()
    #df.account_value.plot()

    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(account_value=df, value_col_name=name)
    perf_stats_all = pd.DataFrame(perf_stats_all)

    # return
    # baseline stats
    print("==============Get Baseline Stats===========")
    baseline_df = get_cached_baseline(
        market_id,
        df.loc[0, "date"],
        df.loc[len(df) - 1, "date"],
    )
    stats = backtest_stats(baseline_df, value_col_name="close")
    print(stats)

    print("==============Compare to baseline===========")
    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    backtest_plot(
        df,
        value_col_name=name,
        baseline_df=baseline_df,
        baseline_start=df.loc[0, "date"],
        baseline_end=df.loc[len(df) - 1, "date"],
    )

def draw_loss(path, time_begin=100, out_dir=Path(".")):
    Q_loss_stat = np.load(path, allow_pickle=True)
    critic_loss = Q_loss_stat.item()["critic_loss"]
    actor_loss = Q_loss_stat.item()["actor_loss"]
    plt.figure(2)
    plt.xlabel("Time",  fontsize=14)
    plt.tick_params(labelsize=10)
    plt.ylabel("Critic loss", fontsize =14)
    plt.title("")
    plt.plot(range(time_begin, len(critic_loss)) ,critic_loss[time_begin:])
    plt.savefig(out_dir/"critic_loss.pdf", format="pdf")

    plt.figure(3)
    plt.xlabel("Time",  fontsize=14)
    plt.tick_params(labelsize=10)
    plt.ylabel("Actor loss", fontsize =14)
    plt.title("")
    plt.plot(range(time_begin, len(actor_loss)), actor_loss[time_begin:])
    plt.savefig(out_dir/"actor_loss.pdf", format="pdf")

if __name__ == "__main__":
    from pathlib import Path
    OUTDIR = Path(os.path.realpath(".") + "/paper")
    df = pd.read_csv("outputs/old/mbpo_dow_2021-09-08_23:45_0_account_value.csv")
    account_value_show(df, 0)
    show_baseline(2, "2018-07-01", "2021-07-01")
    draw_loss("outputs/2021-12-11/mbpo_sp_stats/04:32:25:%f/Q_loss.npy", 300000, OUTDIR)




"""
%matplotlib inline

df_account_value_trial = pd.read_csv("/mnt/wanyao/guiyi/hhf/MBRL4FIN/outputs/mbpo_2021-08-31_11:42_0_account_value.csv")
plot.account_value_show(df_account_value_trial, 2)


%matplotlib inline
import datetime
import os
plt.figure()
for i in range(25):
    startTime = '2021-10-19_21:03'
    time = (datetime.datetime.strptime(startTime, "%Y-%m-%d_%H:%M") + datetime.timedelta(
        minutes = i)).strftime("%Y-%m-%d_%H:%M")
    save_path = f"outputs/mbpo_sp_{time}_0_account_value.csv"
    print(save_path)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        print(i)
        plt.plot(range(len(df.account_value)),df.account_value, label = "i")
        plt.legend('i')#查
#plt.ylim(1,1.5*10**6)                                                                  
plt.show()




%matplotlib inline
import numpy
critic_actor_loss = numpy.load('outputs/mbpo_dow_2021-10-31 23:44:26.982013_Q.npy', allow_pickle=True).item()
critic_loss, actor_loss = critic_actor_loss["critic_loss"], critic_actor_loss["actor_loss"]
plt.figure()
plt.plot(range(len(critic_loss)), critic_loss,color = "r")
#plt.plot(range(len(actor_loss)), actor_loss, color = "b")
plt.show()

"""

"""
#outputs/2021-11-20/mbpo_sp/01:07:25:%f/2021-11-20_09:08_account_value.csv
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
plt.figure()
for i in range(120):
    startTime = '2021-11-20_09:08'
    time = (datetime.datetime.strptime(startTime, "%Y-%m-%d_%H:%M") + datetime.timedelta(
        minutes = i)).strftime("%Y-%m-%d_%H:%M")
        
    save_path = f"outputs/2021-11-20/mbpo_sp/01:07:25:%f/{time}_account_value.csv"
    print(save_path)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        print(save_path)
        print(i)
        plt.plot(range(len(df.account_value)),df.account_value, label = "i")
        plt.legend('i')#查
#plt.ylim(1,1.5*10**6)
plt.show()
"""
