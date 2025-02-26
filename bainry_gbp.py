import MetaTrader5 as mt5
import pandas as pd 
import numpy as np
import pickle
import torch
from torch import nn
from functools import reduce
from torch.serialization import add_safe_globals
import time
import datetime
import threading as th
import pyautogui
import requests
import os


from models import LSTM , CNNLSTM , LSTM_final_model
from data import Data_pross


add_safe_globals([LSTM , CNNLSTM , LSTM_final_model])




def gbp_cnn_lstm():
  model = CNNLSTM(27, 27 ,256, 1 , 1)
  model.load_state_dict(torch.load(r'gbp/cnnlstm_v6.pth',weights_only=True))
  return model


def gbp_lstm():
    input_size = 27
    hidden_size = 256
    num_layers = 3
    bidirectional = False
    num_cls = 1
    batch_first = True
    dropout = 0

    lstm_model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        num_cls=num_cls,
        batch_first=batch_first,
        dropout=dropout
    )


    lstm_model.load_state_dict(torch.load(r'gbp/lstm_v5.pth' ,weights_only=True))
    return lstm_model


def final_model():

    input_size = 539
    hidden_size = 1078
    num_layers = 1
    bidirectional = False
    num_cls = 1
    batch_first = True
    dropout = 0.3

    model = LSTM_final_model(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional = bidirectional ,
        num_cls=num_cls,
        batch_first=batch_first ,
        dropout = dropout
    )
    model.load_state_dict(torch.load(r'gbp/metamodel.pth',weights_only=True))

    return model

        
def gbp_scale():
    
    with open(r'gbp/scale.pkl', 'rb') as f:
        return pickle.load(f)




model_gbp_cnn_lstm = gbp_cnn_lstm()
model_gbp_lstm = gbp_lstm()

 
def pred(models, x ):
    result = []
    for model in models:
        model.eval()
        with torch.no_grad():
        # اطمینان از درست بودن ابعاد ورودی

            # برای LSTM و GRU نیاز به batch_size داریم
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # اضافه کردن بعد batch
            output = model(x)
            result.append(output)


    return torch.cat((result[0] , result[1]), dim=2)


def final(model , x):
        model.eval()
        with torch.no_grad():

            return model(x)

def trades(type):
    if type == 'buy' : 

        pyautogui.keyDown('shift')  # نگه داشتن کلید Shift
        pyautogui.press('w')        # فشار دادن کلید W
        time.sleep(0.1)
        pyautogui.keyUp('shift')
        pyautogui.keyUp('w')
        


    else : 
        pyautogui.keyDown('shift')  # نگه داشتن کلید Shift
        pyautogui.press('s')        # فشار دادن کلید W
        time.sleep(0.1)
        pyautogui.keyUp('shift')
        pyautogui.keyUp('s')
        




def run(symol , scale, models ,gbp_final_model):
    
    while True:
        time.sleep(1)
        if (datetime.datetime.now().minute + 1) % 5 == 0 and(datetime.datetime.now().second + 1) >= 57:
        # if (datetime.datetime.now().second + 1) >= 55:

            # try:
                # if load_df is not None:
                #     new_line = trade.get_data(symbol=symol, n=1)
                #     df = pd.concat([load_df, new_line], ignore_index=True)
                # else:
                # # load_df = df.copy()
                df = trade.df_pross(symol)
                tensor_df = trade.pross_scale(df , scale)

                pred_tensor = pred(models, tensor_df)
                
                main_tensor = torch.cat((tensor_df.unsqueeze(0) , pred_tensor), dim=2)
                final_pred = (final( gbp_final_model ,main_tensor )).detach().item()

                print(final_pred)
                # if final_pred > 0.65 : 
                #     trades('buy')
                # if final_pred < 0.25 : 
                #     trades('sell')

                if final_pred > 0.6 : 
                    trades('buy')
                if final_pred < 0.4 : 
                    trades('sell')


                # print(df)
                time.sleep(10)
            # except Exception as e:
            #     print(f"Error in run: {str(e)}")
            #     print(f"Input shape: {df.shape}")  # چاپ شکل داده ورودی برای دیباگ
            #     time.sleep(10)
            #     continue







login = 225762
pas = 'z$Fh{}3z'
server = 'PropridgeCapitalMarkets-Server'
path = r"C:\Program Files\Propridge Capital Markets MT5 Terminal\terminal64.exe"
symol = 'GBPUSD'




gbp_scaler = gbp_scale()
gbp_final_model = final_model()

trade = Data_pross(login = login , pas = pas , server= server , path = path  )
trade.init()



th.Thread(target= run , args=(  symol , gbp_scaler , [model_gbp_lstm , model_gbp_cnn_lstm  ] , gbp_final_model )).start()












