# 


import numpy as np
import pandas as pd
import os, datetime
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
print('Tensorflow version: {}'.format(tf.__version__))
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pickle
import warnings
warnings.filterwarnings('ignore')

with open('./data/multi_factor_min_max_parameter.pickle', 'rb') as handle:
    normalized_dict = pickle.load(handle)

# load min max parameter
multi_min_return = normalized_dict['multi_min_return']
multi_max_return = normalized_dict['multi_max_return']

vol_max_return = normalized_dict['vol_max_return']
vol_min_return = normalized_dict['vol_min_return']

quote_volume_min_return = normalized_dict['quote_volume_min_return']
quote_volume_max_return = normalized_dict['quote_volume_max_return']

taker_buy_base_asset_volume_max_return = normalized_dict['taker_buy_base_asset_volume_max_return']
taker_buy_base_asset_volume_min_return = normalized_dict['taker_buy_base_asset_volume_min_return']

taker_buy_quote_asset_volume_min_return = normalized_dict['taker_buy_quote_asset_volume_min_return']
taker_buy_quote_asset_volume_max_return = normalized_dict['taker_buy_quote_asset_volume_max_return']

trade_num_min_return = normalized_dict['trade_num_min_return']
trade_num_max_return = normalized_dict['trade_num_max_return']

    

# calculated by original train data
min_return = -0.06512109971782276
max_return = 0.07264907716666236
min_volume = -0.3859540528316058
max_volume = 0.3858981239713559



# min_return = 0.06512109971782276
# max_return = 0.07264907716666236
# min_volume = -0.3859540528316058
# max_volume = 0.3858981239713559
def restore_min_max_scalar(normalized_number):
    original_number =  (normalized_number) * (max_return - min_return)  + min_return
    return original_number

def restore_close_price(input_df, pred_index, pred_number, moving_avg_step=10):
    previous_day_idx = pred_index-1
    # total_df: original df include train & test
    # next_day_index: 
    close_ptc_change =  restore_min_max_scalar(pred_number)
    previois_avg     = input_df.loc[pred_index-moving_avg_step:previous_day_idx,'Close'].mean()
    close_avg        = previois_avg * (close_ptc_change+1)
    restored_close   = (close_avg*moving_avg_step) - (input_df.loc[previous_day_idx- moving_avg_step+2:previous_day_idx, 'Close'].sum())
    return restored_close


class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        '''Initialize weights and biases with shape (batch, seq_len)'''
        self.weights_linear = self.add_weight(name='weight_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

    def call(self, x):
        '''Calculate linear and periodic time features'''
        x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
        time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
        time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)
   
    def get_config(self): # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config
  
class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k, 
                           input_shape=input_shape, 
                           kernel_initializer='glorot_uniform', 
                           bias_initializer='glorot_uniform')

        self.key = Dense(self.d_k, 
                         input_shape=input_shape, 
                         kernel_initializer='glorot_uniform', 
                         bias_initializer='glorot_uniform')

        self.value = Dense(self.d_v, 
                           input_shape=input_shape, 
                           kernel_initializer='glorot_uniform', 
                           bias_initializer='glorot_uniform')

    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out    

#############################################################################

class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
    
        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
        self.linear = Dense(input_shape[0][-1], 
                            input_shape=input_shape, 
                            kernel_initializer='glorot_uniform', 
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear   

#############################################################################

class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
        self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) 
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
  
    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer 

    def get_config(self): # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'ff_dim': self.ff_dim,
                       'attn_heads': self.attn_heads,
                       'dropout_rate': self.dropout_rate})
        return config          


def load_fine_tune_model(weight_path='./model_weight/transformer_btc_multi_factor_ft221.hdf5'):
    model = tf.keras.models.load_model(weight_path,
                                       custom_objects={'Time2Vector': Time2Vector, 
                                                       'SingleAttention': SingleAttention,
                                                       'MultiAttention': MultiAttention,
                                                       'TransformerEncoder': TransformerEncoder})
    return model

def df_normalization(input_df, moving_avg_step=10):
    raw_df = input_df.copy()
    if 'Adj Close' in raw_df.columns:
        del raw_df['Adj Close']
    raw_df[['Open_avg', 'High_avg', 'Low_avg', 'Close_avg', 'Volume_avg']] = raw_df[['Open', 'High', 'Low', 'Close', 'Volume']].rolling(moving_avg_step).mean()
    raw_df.dropna(how='any', axis=0, inplace=True)
    raw_df['Open_ptc_change']   = raw_df['Open_avg'].pct_change() # Create arithmetic returns column
    raw_df['High_ptc_change']   = raw_df['High_avg'].pct_change() # Create arithmetic returns column
    raw_df['Low_ptc_change']    = raw_df['Low_avg'].pct_change() # Create arithmetic returns column
    raw_df['Close_ptc_change']  = raw_df['Close_avg'].pct_change() # Create arithmetic returns column
    raw_df['Volume_ptc_change'] = raw_df['Volume_avg'].pct_change()
    raw_df.dropna(how='any', axis=0, inplace=True)
    raw_df['Open_ptc_change_normalized']  = (raw_df['Open_ptc_change'] - min_return) / (max_return - min_return)
    raw_df['High_ptc_change_normalized']  = (raw_df['High_ptc_change'] - min_return) / (max_return - min_return)
    raw_df['Low_ptc_change_normalized']   = (raw_df['Low_ptc_change'] - min_return) / (max_return - min_return)
    raw_df['Close_ptc_change_normalized'] = (raw_df['Close_ptc_change'] - min_return) / (max_return - min_return)
    raw_df['Volume_ptc_change_normalized'] = (raw_df['Volume_ptc_change'] - min_volume) / (max_volume - min_volume)
    normalized_df = raw_df.reset_index(drop=True)
    return normalized_df

def transformer_prediction_model(normalized_df, predict_last_days=5):
    seq_len=128
    normalized_df = normalized_df.reset_index(drop=True)
    start_time = len(normalized_df) - 128 - predict_last_days
    # select input data range & column
    next_day_input = normalized_df.loc[start_time:,['Open_ptc_change_normalized',
                                                    'High_ptc_change_normalized',
                                                    'Low_ptc_change_normalized',
                                                    'Close_ptc_change_normalized',
                                                    'Volume_ptc_change_normalized']].values
    # convert to model input format
    input_data = []
    for i in range(seq_len, len(next_day_input)):
        input_data.append(next_day_input[i-seq_len:i]) # Chunks of training data with a length of 128 df-rows
    input_data = np.array(input_data)
    # predict the close price
    pred_price_normalized = model.predict(input_data)
    pred_price_normalized = pred_price_normalized.reshape(-1)
    # restore to original pice
    pred_price_list = []
    for i, value in enumerate(pred_price_normalized):
        # 5day end data in -5 index
        normalized_df_index = predict_last_days-i
        pred_price = restore_close_price(normalized_df, normalized_df.index[-int(normalized_df_index)], value,10)
        pred_price_list.append(pred_price)
    # output predicted price and data frame
    output_df = normalized_df.loc[start_time:]
    output_df = output_df.reset_index(drop=True)
    output_df['pred_close'] = ''
    output_df['pred_close_normalized'] = ''
    output_df.loc[len(output_df.index)-len(pred_price_list[0:-1]):,'pred_close'] = pred_price_list[0:-1]
    output_df.loc[len(output_df.index)-len(pred_price_normalized[0:-1]):,'pred_close_normalized'] = pred_price_normalized[0:-1]
    return pred_price_list[-1] ,output_df.loc[len(output_df.index) - predict_last_days:,]



## ----------------------------BTC muliti factor df preprocess-------------------------#

def df_mulit_factor_nomalization(df, moving_avg_step=10):
    BTC_df = df.copy()
    if 'time' in BTC_df.columns:
        del BTC_df['time']
    if 'symbol' in BTC_df.columns:
        del BTC_df['symbol']
    if '下週期幣種漲跌幅' in BTC_df.columns:
        del BTC_df['下週期幣種漲跌幅']
    BTC_df.dropna(how='any', axis=0, inplace=True)
    BTC_df['close_original']  = BTC_df['close']
    # moving average
    BTC_df[['open', 'high', 'low', 'close','volume','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']] = BTC_df[['open', 'high', 'low', 'close','volume','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']].rolling(moving_avg_step).mean()
    # percent change
    BTC_df['open']   = BTC_df['open'].pct_change() # Create arithmetic returns column
    BTC_df['high']   = BTC_df['high'].pct_change() # Create arithmetic returns column
    BTC_df['low']    = BTC_df['low'].pct_change() # Create arithmetic returns column
    # save the close ptc_change for restore the true price
    BTC_df['close']  = BTC_df['close'].pct_change()# Create arithmetic returns column
    BTC_df['volume'] = BTC_df['volume'].pct_change()
    BTC_df['taker_buy_base_asset_volume']  = BTC_df['taker_buy_base_asset_volume'].pct_change()
    BTC_df['taker_buy_quote_asset_volume'] = BTC_df['taker_buy_quote_asset_volume'].pct_change()
    BTC_df.dropna(how='any', axis=0, inplace=True)
    # min_max scalar
    BTC_df['open']         = (BTC_df['open'] - multi_min_return) / (multi_max_return - multi_min_return)
    BTC_df['high']         = (BTC_df['high'] - multi_min_return) / (multi_max_return - multi_min_return)
    BTC_df['low']          = (BTC_df['low'] - multi_min_return) / (multi_max_return - multi_min_return)
    BTC_df['volume']       = (BTC_df['volume'] - vol_min_return) / (vol_max_return - vol_min_return)
    BTC_df['quote_volume'] = (BTC_df['quote_volume'] - quote_volume_min_return) / (quote_volume_max_return - quote_volume_min_return)
    BTC_df['close'] = (BTC_df['close'] - multi_min_return) / (multi_max_return - multi_min_return)
    BTC_df['taker_buy_base_asset_volume'] = (BTC_df['taker_buy_base_asset_volume'] - taker_buy_base_asset_volume_min_return) / (taker_buy_base_asset_volume_max_return -taker_buy_base_asset_volume_min_return)
    BTC_df['taker_buy_quote_asset_volume'] = (BTC_df['taker_buy_base_asset_volume'] - taker_buy_quote_asset_volume_min_return) / (taker_buy_quote_asset_volume_max_return - taker_buy_quote_asset_volume_min_return)
    BTC_df['trade_num'] = (BTC_df['trade_num'] - trade_num_min_return) / (trade_num_max_return - trade_num_min_return)
    return BTC_df


def mulit_factor_restore_min_max_scalar(normalized_number):
    original_number =  (normalized_number) * (multi_max_return - multi_min_return)  + multi_min_return
    return original_number

def mulit_factor_restore_close_price(input_df, pred_index, pred_number, moving_avg_step=10):
    previous_day_idx = pred_index-1
    # total_df: original df include train & test
    # next_day_index: 
    close_ptc_change = mulit_factor_restore_min_max_scalar(pred_number)
    previois_avg     = input_df.loc[pred_index-moving_avg_step:previous_day_idx,'close_original'].mean()
    close_avg        = previois_avg * (close_ptc_change+1)
    restored_close   = (close_avg*moving_avg_step) - (input_df.loc[previous_day_idx- moving_avg_step+2:previous_day_idx, 'close_original'].sum())
    return restored_close

def multi_facter_transformer_prediction_model(model, normalized_df, predict_last_days=5):
    seq_len=128
    if (predict_last_days - 128 ) > len(normalized_df):
        print('enough input data, please reset predict_days')
    normalized_df = normalized_df.reset_index(drop=True)
    start_time = len(normalized_df) - 128 - predict_last_days
    # original close will only be used in restore the true predicted price.
    # normalized close will be the input feature, original feature will be used for restore the true price.
    input_col  = [col for col in list(normalized_df.columns) if col != 'close_original']
    # select input data range & column
    next_day_input = normalized_df.loc[start_time:, input_col].values
    # convert to model input format
    input_data = []
    for i in range(seq_len, len(next_day_input)):
        input_data.append(next_day_input[i-seq_len:i]) # Chunks of training data with a length of 128 df-rows
    input_data = np.array(input_data)
    # predict the close price
    pred_price_normalized = model.predict(input_data)
    pred_price_normalized = pred_price_normalized.reshape(-1)
    # restore to original pice
    pred_price_list = []
    for i, value in enumerate(pred_price_normalized):
        # 5day end data in -5 index
        normalized_df_index = predict_last_days-i
        pred_price = mulit_factor_restore_close_price(normalized_df, normalized_df.index[-int(normalized_df_index)], value,10)
        pred_price_list.append(pred_price)
    # output predicted price and data frame
    output_df = normalized_df.loc[start_time:]
    output_df = output_df.reset_index(drop=True)
    output_df['pred_close'] = ''
    output_df['pred_close_normalized'] = ''
    output_df.loc[len(output_df.index)-len(pred_price_list[0:-1]):,'pred_close'] = pred_price_list[0:-1]
    output_df.loc[len(output_df.index)-len(pred_price_normalized[0:-1]):,'pred_close_normalized'] = pred_price_normalized[0:-1]
    return output_df