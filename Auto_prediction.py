import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import numpy as np
import pandas as pd
import ccxt
import talib
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from script.fetch_history_data import binance_fetch_history_price, binance_single_fetch_history_price
from script.preprocess import *
from sklearn.preprocessing import StandardScaler
from script.transformer_timestep import *
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle
import schedule

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

with open('multi_factor_v2.0.2.pickle', 'rb') as input_pickle:
    multi_factor_dic = pickle.load(input_pickle)

zscore_col = multi_factor_dic['zscore'][0]
zscore_std = multi_factor_dic['zscore'][1]
zscore_mean = multi_factor_dic['zscore'][2]
input_col = multi_factor_dic['input_col']

batch_size = 32
seq_len = 128
d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256
valid_percent = 20
test_percent = 10

target_column='Close_MA_ptc_10'
pearson_corr_cutoff = 0.2

def calculate_z_score_input_format(df_cal):
    for col in zscore_col:
        df_cal[col] = (df_cal[col] - zscore_mean) / zscore_std
    return df_cal[input_col], df_cal

def restore_single_zscore_close_price_v3(test_pred, df_zscore_total, seq_len=128):
    single_df = df_zscore_total.reset_index(drop=True).copy()
    # df_zscore_output['pred_Close']='-'
    # # 128~len(df)
    # for i in range(seq_len, len(df_zscore_output)):
    #     # restore the model predicted_ptc
    pred_ptc = test_pred[0][0]*zscore_std + zscore_mean
    # choose the single_df[0:128] (index 0~127) 
    # take the final MA_10
    pre_MA = single_df['Close_MA_10'][single_df.index[-1]]
    # final MA_10* pred_ptc = pred_MA10
    close_MA = pre_MA*(pred_ptc+1)
    # pred_MA_10*10 - pre 9 Close = pred_Close
    pred_close = close_MA*10 - single_df['Close'][-9:].sum()
    return pred_close


def predict_single_next_price():
    df = binance_single_fetch_history_price(coin='BTC/USDT', timeframe='1h', seq_len=seq_len)
    timeframes = [10,20,40,60,80,100,120,140,160,180,200]
    df_cal = calculate_technical_indicators(df, timeframes=timeframes)
    df_zscore, df_zscore_total = calculate_z_score_input_format(df_cal)
    X_test = preprocess_single_sequence_data(df_zscore, seq_len=seq_len)
    pred_test = model.predict(X_test)
    pred_close = restore_single_zscore_close_price_v3(pred_test, df_zscore_total)
    #output_df = df_zscore_total.loc[327, df_zscore_total.columns[:6]]
    return pred_close

def prediction_log(pred_close):
    df = binance_single_fetch_history_price(coin='BTC/USDT', timeframe='1h', seq_len=seq_len)
    timeframes = [10,20,40,60,80,100,120,140,160,180,200]
    df_cal = calculate_technical_indicators(df, timeframes=timeframes)
    df_zscore, df_zscore_total = calculate_z_score_input_format(df_cal)
    X_test = preprocess_single_sequence_data(df_zscore, seq_len=seq_len)
    pred_test = model.predict(X_test)
    next_pred_close = restore_single_zscore_close_price_v3(pred_test, df_zscore_total)
    output_df = df_zscore_total.loc[327, df_zscore_total.columns[:6]]
    output_df['Prediction_Close'] = pred_close
    output_df['Next_prediction_Close'] = next_pred_close
    save_df = pd.DataFrame(output_df).T
    log_file = pd.read_excel('./prediction_log/predcition_log.xlsx')
    log_file = log_file.append(save_df, ignore_index=True)
    log_file.to_excel('./prediction_log/predcition_log.xlsx', index=False)
    print('---------------------------------------------------')
    print('TimeStamp-----------------------: ', save_df['Timestamp'][327])
    print('Prediction Close Price----------: ', pred_close)
    print('Actually Close Price------------: ', save_df['Close'][327])
    print('Next Hour Prediction Close Price: ', save_df['Next_prediction_Close'][327])
    return next_pred_close

def next_hour_calculator():
    now = datetime.datetime.now()
    next_hour = now.replace(hour=now.hour+1, minute=0, second=0, microsecond=0)
    delta = next_hour - now
    seconds_left = delta.seconds
    return seconds_left

if __name__ == '__main__':
    print('Loading Model...')
    model = tf.keras.models.load_model('./model_weight/transformer_btc_multi_factor_v2.0.2.hdf5',
                                   custom_objects={'Time2Vector': Time2Vector, 
                                                   'SingleAttention': SingleAttention,
                                                   'MultiAttention': MultiAttention,
                                                   'TransformerEncoder': TransformerEncoder})
    # print this hour prediction price
    df = binance_single_fetch_history_price(coin='BTC/USDT', timeframe='1h', seq_len=seq_len)
    timeframes = [10,20,40,60,80,100,120,140,160,180,200]
    df_cal = calculate_technical_indicators(df, timeframes=timeframes)
    Timestamp = df_cal['Timestamp'][df_cal.index[-1]]
    Open_price = df_cal['Close'][df_cal.index[-1]]
    pred_close = predict_single_next_price()
    print('---------------------------------------------------')
    print('TimeStamp-----------------------: ', Timestamp)
    print('This Hour Open Price------------: ', Open_price)
    print('This Hour Prediction Close Price: ', pred_close)
    print(f"Next prediction will start after {next_hour_calculator()} seconds......")
    #time.sleep(1)
    time.sleep(next_hour_calculator())
    while True:
        next_pred_close = prediction_log(pred_close)
        # fetch data not be updated yet
        while pred_close == next_pred_close:
            next_pred_close = prediction_log(pred_close)
        pred_close = next_pred_close
        #time.sleep(2)
        time.sleep(3600)
