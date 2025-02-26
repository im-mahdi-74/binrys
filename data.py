
import MetaTrader5 as mt5
import pandas as pd 
import numpy as np
from functools import reduce
import time
import datetime
import torch


class Data_pross:

    def __init__(self , login , pas , server ,  path  ) : 
        self.login = login 
        self.pas = pas
        self.server = server 
        self.path = path 

    def init(self):

        init = mt5.initialize(path=self.path)
        login = mt5.login(self.login, password=self.pas , server=self.server)
        return init , login
    
    def copy_rates_from_pos(self , symbol , timeframe , n = None ):
        if n == None :
            n = 200
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC')
        df['time'] = df['time'].dt.tz_localize(None)
        df.drop(['tick_volume' , 'spread' , 'real_volume'] , inplace=True , axis=1)
        return df


    def df_pross(self , symbol):
       
        df_5  = self.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5 ,)
        df_15 = self.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15,)
        df_30 = self.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30,)
        df_1h = self.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1 ,)
        df_4h = self.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4 ,)


        df_15.rename(columns={'open': 'open15',  'close': 'close15' , 'high' : 'high15' , 'low' : 'low15'}, inplace=True)
        df_30.rename(columns={'open': 'open30',  'close': 'close30' , 'high' : 'high30' , 'low' : 'low30'}, inplace=True)
        df_1h.rename(columns={'open': 'open1h',  'close': 'close1h' , 'high' : 'high1h' , 'low' : 'low1h'}, inplace=True)
        df_4h.rename(columns={'open': 'open4h',  'close': 'close4h' , 'high' : 'high4h' , 'low' : 'low4h'}, inplace=True)

        df_5.set_index('time', inplace=True)
        df_15.set_index('time', inplace=True)
        df_30.set_index('time', inplace=True)
        df_1h.set_index('time', inplace=True)
        df_4h.set_index('time', inplace=True)

        # ادغام بر اساس ایندکس (زمان)
        merged = df_5.join([df_15, df_30, df_1h, df_4h], how='outer').sort_index()


        # df.reset_index(drop=True , inplace= True)
        merged.reset_index(inplace= True)
        merged.ffill(inplace=True)
        merged['5']  = np.where(merged['close']   - merged['open'] > 0 , 1 , -1)
        merged['15'] = np.where(merged['close15'] - merged['open15'] > 0 , 1 , -1)
        merged['30'] = np.where(merged['close30'] - merged['open30'] > 0 , 1 , -1)
        merged['H1'] = np.where(merged['close1h'] - merged['open1h'] > 0 , 1 , -1)
        merged['H4'] = np.where(merged['close4h'] - merged['open4h'] > 0 , 1 , -1)
        merged['hour'] = merged['time'].dt.hour
        merged['minute'] = merged['time'].dt.minute
        merged = merged[merged['time'].dt.hour >= 7]
        merged = merged[merged['time'].dt.hour <= 22]
        merged.dropna(inplace=True)
        merged.reset_index(inplace= True , drop= True)
        merged.drop('time' , axis = 1 , inplace = True)
        merged = merged.iloc[-27:].reset_index(drop=True)
        return merged
                

    def pross_scale(self, df ,  scaler):
  
        x = scaler.transform(np.array(df))

        return  torch.tensor(x, dtype=torch.float32)
        
