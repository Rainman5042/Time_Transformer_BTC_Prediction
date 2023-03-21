"""
development python version:3.8.13
author: Siao-Yu Jian
function: generate the technical_indicators for btc dataframe
input: btc price dataframe, timeframs list
output: Dataframe with a lot of indicators base on different timeframes
version history:
                2023.03.10      version v1.0.0      initial version
"""
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime
import talib
from tqdm import tqdm
"""
function: generate all technical_indicators
input: Dataframe, Timeframe
output: a lot of indicators with different timeframes
"""
def ZhangDieFu_label(x):
    if abs(x) <= 0.0005:
        return 0
    elif x > 0.0005:
        return 1
    else:
        return 2

def calculate_technical_indicators(input_df, timeframes=[5,10,20,40,60,80,100,120,140,160,180,200]):
    df = input_df.copy()
    df = df.sort_values(by=['Timestamp']).reset_index(drop=True)
    
    for timeframe in timeframes:
        df['PrevClose'] = df['Close'].shift(1)
        df['PrevVolume'] = df['Volume'].shift(1)
        
        # MA
        df[f'open_MA_{timeframe}'] = input_df['Open'].rolling(timeframe).mean()
        df[f'High_MA_{timeframe}'] = input_df['High'].rolling(timeframe).mean()
        df[f'Low_MA_{timeframe}'] = input_df['Low'].rolling(timeframe).mean()
        df[f'Close_MA_{timeframe}'] = input_df['Close'].rolling(timeframe).mean()
        df[f'Volume_MA_{timeframe}'] = input_df['Volume'].rolling(timeframe).mean()
        # MA_ptc_change
        df[f'open_MA_ptc_{timeframe}'] = df[f'open_MA_{timeframe}'].pct_change() 
        df[f'High_MA_ptc_{timeframe}'] = df[f'High_MA_{timeframe}'].pct_change() 
        df[f'Low_MA_ptc_{timeframe}'] = df[f'Low_MA_{timeframe}'].pct_change() 
        df[f'Close_MA_ptc_{timeframe}'] = df[f'Close_MA_{timeframe}'].pct_change() 
        df[f'Volume_MA_ptc_{timeframe}'] = df[f'Volume_MA_{timeframe}'].pct_change() 
        
        # Bias
        df['Bias'] = (df['Close'] - talib.SMA(df['Close'], timeperiod=timeframe)) / talib.SMA(df['Close'], timeperiod=timeframe) * 100

        # Cmo
        df[f'Cmo_{timeframe}'] = talib.CMO(df['Close'], timeperiod=timeframe)

        # Atr
        df[f'Atr_{timeframe}'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=timeframe)

        # Cci
        df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['MeanDeviation'] = abs(df['TypicalPrice'] - talib.SMA(df['TypicalPrice'], timeperiod=timeframe))
        df[f'Cci_{timeframe}'] = (df['TypicalPrice'] - talib.SMA(df['TypicalPrice'], timeperiod=timeframe)) / (0.015 * df['MeanDeviation'])

        # Volume
        df['Volume_pct'] = df['Volume'] / df['PrevVolume'] - 1

        ## Psy
        #df[f'PSY_timeframe_{timeframe}'] = talib.PSY(df['Close'], timeperiod=timeframe) / 100

        # ZhangDieFu
        df['ZhangDieFu'] = (df['Close'] - df['PrevClose']) / df['PrevClose']

        # ZhenFu
        df['ZhenFu'] = (df['High'] - df['Low']) / df['PrevClose']

        # Rsi
        df[f'Rsi_{timeframe}'] = talib.RSI(df['Close'], timeperiod=timeframe)

        # Ic
        df[f'Ic_{timeframe}'] = (df['Close'] - talib.EMA(df['Close'], timeperiod=timeframe)) / df['Close']

        # MACD
        df['Macd'], df['Signal'], df['Histogram'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        # Bollinger Bands
        df[f'UpperBB_{timeframe}'], df[f'MiddleBB_{timeframe}'], df[f'LowerBB_{timeframe}'] = talib.BBANDS(df['Close'], timeperiod=timeframe, nbdevup=2, nbdevdn=2, matype=0)

        # Stochastic Oscillator
        df['SlowK'], df['SlowD'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

        # Moving Average Envelope
        df[f'UpperMAE_{timeframe}'], df[f'MiddleMAE_{timeframe}'], df[f'LowerMAE_{timeframe}'] = talib.MA(df['Close'], timeperiod=timeframe), talib.MA(df['Close'], timeperiod=timeframe), talib.MA(df['Close'], timeperiod=timeframe)
        df[f'UpperMAE_{timeframe}'] += talib.STDDEV(df['Close'], timeperiod=timeframe) * 2
        df[f'LowerMAE_{timeframe}'] -= talib.STDDEV(df['Close'], timeperiod=timeframe) * 2

        # On Balance Volume
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])

        # Remove intermediate columns
        df.drop(['PrevClose', 'PrevVolume', 'TypicalPrice', 'MeanDeviation'], axis=1, inplace=True)
    df = df.dropna()
    return df



"""
function: create the mean and std dictionary for calculate Z-score
input: Dataframe
output: dic['column'] = [mean, std]
"""
def z_score_dic(input_df):
    dic = {}
    for col in input_df.columns[1:]: # ignore timestamp
        dic[col] = [input_df[col].mean(), input_df[col].std()]
    return dic

"""
function: calculate Z-score
input: Dataframe
output: normalize to Z-score dataframe
"""
def calculate_z_score(input_df, dic):
    df = input_df.copy()
    for col in df.columns[1:]: # ignore timestamp
        mean, std = dic[col][0], dic[col][1]
        df[col] = (df[col] - mean) / std
    return df

"""
function: recover z_score to original value
input: normalized Z-score dataframe, z_score_dic
output: original Dataframe
"""
def recover_z_score(input_df, dic):
    df = input_df.copy()
    for col in df.columns[1:]: # ignore timestamp
        mean, std = dic[col][0], dic[col][1]
        df[col] = (df[col] + mean) * std
    return df

"""
function: create the max and min dictionary for MaxMinScalar
input: Dataframe
output: dic['column'] = [max, min]
"""
def min_max_dic(input_df):
    dic = {}
    for col in input_df.columns[1:]: # open high low close
        dic[col] = [input_df[col].max(), input_df[col].min()]
    return dic

"""
function: calculate MinMaxScalar
input: Dataframe
output: dic['column'] = [max, min]
"""
def calculate_min_max(input_df, dic):
    df = input_df.copy()
    for col in df.columns[1:]: # open high low close
        max_, min_ = dic[col][0], dic[col][1]
        df[col] = (df[col] - min_) / (max_-min_)
    return df



"""
function: recover z_score to original value
input: normalized Z-score dataframe, z_score_dic
output: original Dataframe
"""
def preprocess_train_test_split(input_df, valid_percent=20, test_percent=10):
    df = input_df.copy()
    if 'Timestamp' in df.columns:
        del df['Timestamp']
    last_10pct = sorted(df.index.values)[-int(test_percent/100*len(df))] # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(valid_percent/100*len(df))] # Last 20% of series
    df_train = df[(df.index < last_10pct)]  # Training data are 80% of total data
    df_val   = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    df_test  = df[(df.index >= last_10pct)]
    return df_train, df_val, df_test


"""
function: transfer data to sequence format for model training
input: splited_dataframe, target_column ,sequence_len
output: sequence
"""
def preprocess_sequence_data(input_df, target_column='ZhangDieFu', seq_len=128):
    X_train, y_train = [], []
    df_value = input_df.values
    df_target_value = input_df[target_column].values
    for i in range(seq_len, len(df_value)):
        X_train.append(df_value[i-seq_len:i]) # Chunks of training data with a length of 128 df-rows
        y_train.append(df_target_value[i]) #Value of 4th column (Close Price) of df-row 128+1
    return np.array(X_train), np.array(y_train)



def preprocess_single_sequence_data(input_df, seq_len=128):
    X_train= []
    df_value = input_df.values
    X_train.append(df_value[:seq_len]) # Chunks of training data with a length of 128 df-rows
    return np.array(X_train)