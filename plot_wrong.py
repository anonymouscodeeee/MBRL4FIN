import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
import pathlib

current_dir = pathlib.Path(os.path.realpath("."))
datasets_dir = current_dir / "outputs"

cache_paths = [datasets_dir/"baseline_dow.csv", datasets_dir/"baseline_nas.csv", datasets_dir/"basseline_sp.csv"]

from finrl.trade.backtest import (
    backtest_stats,
    backtest_plot,
    get_daily_return,
    get_baseline,
)

def account_value_show(df, choose_num=0):
    # plt.figure(trial)
    baseline_tickers = ["^DJI", "^NDX", "^GSPC"]
    df.account_value.plot()

    print("==============Get Backtest Results===========")
    now = datetime.now().strftime("%Y%m%d-%Hh%M")
    perf_stats_all = backtest_stats(account_value=df)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    
    #return
    # baseline stats
    print("==============Get Baseline Stats===========")
    baseline_path = cache_paths[choose_num]
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
    else:
        baseline_df = get_baseline(
            ticker=baseline_tickers[choose_num],
            start=df.loc[0, "date"],
            end=df.loc[len(df) - 1, "date"],
        )

    stats = backtest_stats(baseline_df, value_col_name="close")

    print("==============Compare to DJIA===========")
    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    backtest_plot(
        df,
        baseline_ticker=baseline_tickers[choose_num],
        baseline_start=df.loc[0, "date"],
        baseline_end=df.loc[
            len(df) - 1, "date"
        ],
    )


%matplotlib inline
import datetime
import os
for i in range(60):
    startTime = '2021-10-23_00:00'
    time = (datetime.datetime.strptime(startTime, "%Y-%m-%d_%H:%M") + datetime.timedelta(
        minutes = i)).strftime("%Y-%m-%d_%H:%M")
    save_path = f"outputs/mbpo_nas_{startTime}_0_account_value.csv"
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        df.account_value.plot()