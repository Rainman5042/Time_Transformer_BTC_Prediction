"""
development python version:3.8.13
author: Siao-Yu Jian
function: fetch the btc history data from Binance
input: coin type, timeframe, start_date
output: all history data with Dataframe type, header = [Timestamp, 'Open', 'High', 'Low', 'Close', Volune]
version history:
                2023.03.10      version v1.0.0      initial version
                2023.03.16      version v1.0.1      drop the lastest history data (floating k bar)
                2023.03.22                          let binance api key can load from json file
"""

import numpy as np
import pandas as pd
import ccxt
from datetime import datetime,timezone,timedelta
import talib
import pickle
import json

with open('binance_api_key.json', 'r') as f:
    binance_api_key = json.load(f)

# binanace api setting
binance_exchange = ccxt.binance({
    'apiKey': binance_api_key['api_key'],
    'secret': binance_api_key['api_secret'],
    'options': {
        'defaultType': 'future',},
})



"""
function: convert day to timestamp
input: yyyy-mm-dd
output: int(timestamp)
"""
def convert_day_to_timestamp(day):
    # dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    # dt2 = dt1.astimezone(timezone(timedelta(hours=8)))
    startDate = day
    startDate = datetime.strptime(startDate, "%Y-%m-%d")
    startDate = datetime.timestamp(startDate)
    startDate = int(startDate) * 1000
    return startDate


"""
function: convert timestamp to day
input: int(timestamp)
output: yyyy-mm-dd
"""
def convert_timestamp_to_day(timestamp):
    # dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    # dt2 = dt1.astimezone(timezone(timedelta(hours=8)))
    return datetime.fromtimestamp(timestamp/1000).strftime("%Y-%m-%d-%H:%M")


"""
function: fetch all USD future(contrat) history price from binance
input: coin type, timeframe, start_date
output: all history data with Dataframe type, header = [Timestamp, 'Open', 'High', 'Low', 'Close', Volune]
"""
def binance_fetch_history_price(coin='BTC/USDT', timeframe='8h', start_date='2019-09-09'):
    limit_count = 500
    start_ohlc = binance_exchange.fetch_ohlcv(coin, timeframe, since=convert_day_to_timestamp(start_date), limit=500)
    # fetch 500 data each time
    while limit_count == 500:
        start_time = start_ohlc[-1][0]
        ohlc = binance_exchange.fetch_ohlcv(coin, timeframe, since=start_time, limit=500)
        limit_count = len(ohlc)
        start_ohlc += ohlc
    header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(start_ohlc, columns=header)
    df['Timestamp'] = df['Timestamp'].apply(lambda x:convert_timestamp_to_day(x))
    df = df[:-1] # drop the lastest floating K bar
    return df

"""
function: fetch lastest hour USD future(contrat) history price from binance
input: coin type, timeframe, seq_len=128
output: lastest hour history data with Dataframe type, header = [Timestamp, 'Open', 'High', 'Low', 'Close', Volune]
"""
def binance_single_fetch_history_price(coin='BTC/USDT', timeframe='1h', seq_len=128):
    # 200 for indicator, 1 for drop lastest floating price, 1 for restore close price
    limit_count = seq_len+201
    start_ohlc = binance_exchange.fetch_ohlcv(coin, timeframe, limit=limit_count)
    header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(start_ohlc, columns=header)
    df['Timestamp'] = df['Timestamp'].apply(lambda x:convert_timestamp_to_day(x))
    return df[:-1]
