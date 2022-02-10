"""Contains methods and classes to collect data from
Yahoo Finance API
"""

import pandas as pd
import yfinance as yf
import config
# yfinance接口在中国现在使用不了，只能自己下载到缓存之后改yfinance的代码
# envs\py372\Lib\site-packages\yfinance\base.py:196
# load from cache
"""
        import json
        with open("C:\\Users\\Forever\\Desktop\\mbrl4fin\\data\\baseline_sp.json") as f:
            data = json.loads(f.read())
"""

from finrl.plot import get_baseline
from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
import time
import urllib.parse


def construct_url(tick, start_date, end_date, intervale="1d"):
    # intervale: 1d
    # start_date, end_date: "2000-01-01"
    tick = urllib.parse.quote(tick)
    period1 = int(time.mktime(time.strptime(str(start_date), "%Y-%m-%d")))
    period2 = int(time.mktime(time.strptime(str(end_date), "%Y-%m-%d")))
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{tick}?period1={period1}&period2={period2}&interval={intervale}&includePrePost=False&events=div%2Csplits"
    return url


baseline_ticks = ["^DJI", "^NDX", "^GSPC"]

dji = YahooDownloader(
    start_date="2000-01-01", end_date="2021-11-01", ticker_list=["^GSPC"]
).fetch_data()
