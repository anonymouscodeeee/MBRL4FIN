import pandas as pd
from datetime import datetime
import sys,os
from pyfolio import timeseries

from finrl.plot import (
    backtest_stats,
    backtest_plot,
    get_daily_return,
    get_baseline,
)

def stat_all(df, value_col_name="account_value"):
    dr_test = get_daily_return(df, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )

    return perf_stats_all