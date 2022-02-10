import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta  # 记录outputs_time，记录循环用时

# matplotlib.use('Agg'
from IPython import display

#%matplotlib inline
import torch
from pprint import pprint
import itertools
from pathlib import Path

from stable_baselines3 import A2C

import sys, os
import hashlib


from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer

#%load_ext autoreload
#%autoreload 2

mpl.rcParams.update({"font.size": 16})


# import sys
# sys.path.append("../FinRL-Library")

# import sys,os
# sys.path.append(os.path.dirname(os.path.realpath(".")))
import yfinance as yf


DATASETS_FULL_PATH = [
    "dow_full.csv",
    "nas_full.csv",
    "sp_full.csv",
]


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data


def preprocess(
    dataset_dir,
    market_id,
    start_date,
    end_date,
    ticker_list,
    train_start,
    train_end,
    val_start,
    val_end,
    test_start,
    test_end,
    tech_indicators,
    cache_dir,
):
    ticker_list.sort() 
    encoder = hashlib.sha256()
    encoder.update("_".join(list(ticker_list)).encode())
    encoder.update("_".join(list(tech_indicators)).encode())
    cache_path = cache_dir/ f"data_{market_id}_{start_date}_{end_date}_{encoder.hexdigest()}.csv"
    # 缓存原始数据
    if os.path.exists(cache_path):
        processed_full = pd.read_csv(cache_path)
        print(f"load data from cahe: {cache_path} .")
    else:
        """
        df = YahooDownloader(
            start_date,  #'2000-01-01',
            end_date,  # 2021-01-01，预计将改日期改为'2021-06-20'（今日日期）
            ticker_list=ticker_list,
        ).fetch_data()  # DOW_30_TICKER)道琼斯30只股票

        assert len(df["tic"].unique().tolist()) == len(ticker_list)
        """
        # load raw data from cache file
        df = pd.read_csv(dataset_dir / DATASETS_FULL_PATH[market_id])
        df = df[
            (df.date >= start_date) & (df.date < end_date) & df.tic.isin(ticker_list)
        ]
        # 数据预处理###############################
        df.sort_values(["date", "tic"]).head()

        # tech_indicators = ["macd", "rsi_30", "cci_30", "dx_30"]

        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=tech_indicators,
            use_turbulence=False,
            user_defined_feature=False,
        )
        processed = fe.preprocess_data(df)

        list_date = list(
            pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
        )  # 成一个固定频率的时间索引
        combination = list(itertools.product(list_date, ticker_list))
        """
        1.pandas.date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None, **kwargs)
        由于import pandas as pd,所以也可以写成pd.date_range（start=None, end=None）
        该函数主要用于生成一个固定频率的时间索引，使用时必须指定start、end、periods中的两个参数值，否则报错。
        2.df.astype('str') #改变整个df变成str数据类型
        3.itertools.product(*iterables[, repeat]) # 对应有序的重复抽样过程
        itertools.product(a,b),将a,b元组中的每个分量依次乘开。
        """

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
            processed, on=["date", "tic"], how="left"
        )
        """1.  pd.DataFrame( 某数据集 ，index  ，columns ),给某数据集加上行名index和列名columns
            此处只有pd.DataFrame( 某数据集 ，columns )，第一列加列名date，第二列加列名tic.
        2.  merge(df1,df2,on='key',how)
        按照["date","tic"]为关键字链接，以左边的dataframe为主导，左侧dataframe取全部数据，右侧dataframe配合左边
        """

        processed_full = processed_full[processed_full["date"].isin(processed["date"])]
        # isin函数，清洗数据，删选过滤掉processed_full中一些行，processed_full新加一列['date']若和processed_full中的['date']不相符合，则被剔除
        processed_full = processed_full.sort_values(["date", "tic"])

        processed_full = processed_full.fillna(0)

        processed_full.index = processed_full.date.factorize()[0]

        processed_full.to_csv(cache_path, index=False)

    print(processed_full.columns)
    # 对于processed_full数据集中的缺失值使用 0 来填充.
    processed_full.sample(5)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    train_period = (train_start, train_end)
    val_period = (val_start, val_end)
    unique_trade_date = (test_start, test_end)
    full_period = (train_start, test_end)
    # unique_trade_date = df[(df.datadate > 20151001)&(df.datadate <= 20200720)].datadate.unique()


    processed_full = data_split(processed_full, train_start, test_end)
    df_train = data_split(processed_full, train_start, train_end)
    df_val = data_split(processed_full, val_start, val_end)
    df_test = data_split(processed_full, test_start, test_end)
    
    print(f"train_date:{train_period}, days:{len(df_train.date.unique())}")
    print(f"val_date:{val_period}, days:{len(df_val.date.unique())}")
    print(f"test_date:{unique_trade_date}, days:{len(df_test.date.unique())}")
    print("full_period", full_period)

    assert len(df_train["tic"].unique().tolist()) == len(ticker_list)
    assert len(df_val["tic"].unique().tolist()) == len(ticker_list)
    assert len(df_test["tic"].unique().tolist()) == len(ticker_list)

    return processed_full, df_train, df_val, df_test


# 由于df中的最后一只股票"V"仅有3221天数据，其他29支股票均有5248天数据，故现将其padding为5248天数据
# 使用每一只股票数据与第一只AAPL长度比较
def data_extra(data, col_name, ticker_name, device):
    extra_data = [data[data["tic"] == tic][col_name].to_list() for tic in ticker_name]
    for i, price in enumerate(extra_data):
        if len(price) != len(extra_data[0]):
            extra_data[i] = [0] * (len(extra_data[0]) - len(price)) + extra_data[i]
            print(ticker_name[i], len(price), len(extra_data[i]))
    extra_data = torch.tensor(extra_data, device=device)
    return extra_data


def tick_pick(
    dataset_dir: Path, market_id, ticks, num=30, duration=180, ascending=True
):
    # 根据换手率来选择股票
    if num >= len(ticks):
        return ticks

    data_dates = pd.read_csv(dataset_dir / DATASETS_FULL_PATH[market_id])
    groups = data_dates.groupby(["tic"])
    total_volumes = np.zeros((len(groups),))
    for i, (tic, data) in enumerate(groups):
        if tic in ticks:
            data = data.sort_values(["date"], ascending=False)
            total_volumes[i] = data.volume.iloc[0:duration].sum()
    ticks_profile = pd.read_csv(dataset_dir / "stock_profile.csv")
    ticks_profile.drop_duplicates(subset=["tic"], keep="first", inplace=True)
    all_ticks = data_dates.tic.unique().tolist()
    all_ticks.sort()
    tmp_df = pd.DataFrame(
        {
            "tic": all_ticks,
            "total_volume": total_volumes.tolist(),
        },
    )

    res = pd.merge(tmp_df, ticks_profile, on="tic", how="inner")
    res["exchange rate"] = res["total_volume"] / res["Total issuance"]
    # 选择换手率最低的股票
    chosen_tics = (
        res[["tic", "exchange rate"]]
        .sort_values(["exchange rate"], ascending=ascending)
        .tic[:num]
        .tolist()
    )

    return chosen_tics


if __name__ == "__main__":
    pass
