"""
择时：RSRS择时
持仓：有开仓信号时持有10只股票，不满足时保持空仓

"""
# 导入函数库
import statsmodels.api as sm
import pandas as pd
import pathlib
import sys, os
import torch
import numpy as np

current_dir = pathlib.Path(os.path.realpath("."))


def init_rsrs(choose_num, ticker_list, date_start, date_end, rs_N):
    market_name = ["dow_full.csv", "nas_full.csv", "sp_full.csv"]
    data_full = pd.read_csv(current_dir / f"datasets/{market_name[choose_num]}")
    # data_full.index = data_full.date.factorize()[0]
    data_full = data_full[
        (data_full.date >= date_start)
        & (data_full.date < date_end)
        & (data_full.tic.isin(ticker_list))
    ]
    data_full = data_full.sort_values(["tic", "date"], ignore_index=True)
    # 用于记录回归后的beta值，即斜率
    ans = []
    # 存储计算被决定系数加权修正后的贝塔值
    ans_rightdev = []

    for tic in ticker_list:
        prices = data_full[data_full.tic == tic][["high", "low"]]

        high = prices.high.tolist()
        low = prices.low.tolist()
        ans.append([])
        ans_rightdev.append([])
        for j in range(len(high))[rs_N :]:
            data_high = high[j - rs_N : j]
            data_low = low[j - rs_N : j]
            X = sm.add_constant(data_low)
            model = sm.OLS(data_high, X)
            results = model.fit()
            ans[-1].append(results.params[1])
            # 计算r2
            ans_rightdev[-1].append(results.rsquared)
    return ans, ans_rightdev


def update(date, ans_list, ans_rightdev_list, ticker_list, choose_num, rs_M, rs_N):
    rsrs_rightdev = []
    market_name = ["dow_full.csv", "nas_full.csv", "sp_full.csv"]
    data_full = pd.read_csv(current_dir / f"datasets/{market_name[choose_num]}")
    # data_full.index = data_full.date.factorize()[0]
    data_full = data_full[(data_full.date < date)]
    for i, security in enumerate(ticker_list):
        # 填入各个日期的RSRS斜率值
        beta = 0
        r2 = 0
        # RSRS斜率指标定义
        prices = data_full[data_full.tic == security][["high", "low"]]

        #有可能当前日期之前没有数据，则用0填充
        if len(prices) == 0 :
            ans_list[i].append(0)
            ans_rightdev_list[i].append(0)
            rsrs_rightdev.append(0)
            continue

        # prices = data_full.sort_values(["tic", "date"], ignore_index=True)
        # 选出今天之前N天数据
        highs = prices.iloc[-rs_N:].high.tolist()
        lows = prices.iloc[-rs_N:].low.tolist()              

        X = sm.add_constant(lows)
        model = sm.OLS(highs, X)
        t = model.fit()
        #当拟合的时候斜率是0的时候，params只存放了常数项
        beta = t.params[1] if len(t.params) == 2 else 0
        ans_list[i].append(beta)
        # 计算r2
        r2 = t.rsquared
        ans_rightdev_list[i].append(r2)

        # 计算标准化的RSRS指标
        # 计算均值序列
        section = ans_list[i][-rs_M :]
        # 计算均值序列
        mu = np.mean(section)
        # 计算标准化RSRS指标序列
        sigma = np.std(section)+1e-8
        zscore = (section[-1] - mu) / sigma
        # 计算右偏RSRS标准分
        zscore_rightdev = zscore * beta * r2
        rsrs_rightdev.append(zscore_rightdev)

    return rsrs_rightdev

