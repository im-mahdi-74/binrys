import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
from torch.utils.data import TensorDataset , DataLoader
import pickle

from sklearn.preprocessing import StandardScaler as SC
from sklearn.model_selection import train_test_split
from functools import reduce

from torch.serialization import add_safe_globals
from models import LSTM 
add_safe_globals([LSTM ])

# def gbp_lstm():
input_size = 29
hidden_size = 256
num_layers = 1
bidirectional = False
num_cls = 1
batch_first = True
dropout = 0

model = LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional = bidirectional ,
    num_cls=num_cls,
    batch_first=batch_first ,
    dropout = dropout
)

model.load_state_dict(torch.load(r'D:\env\torch_dj_5min_bainry\wight\fuck_lstm.pth' , weights_only=True))


class GBP():

    def __init__(self ):
        pass
        
    def pros_test(self , df , scale): 

        x_train = np.array(df)
        x_train = scale.transform(x_train)
        tensor_x_train = torch.tensor([x_train], dtype=torch.float32)
        return tensor_x_train


    def gbp_scale(self ):
        
        with open(r'fuck.pkl', 'rb') as f:
            return pickle.load(f)
            




    def pred(self , model , x):
        outputs = model(x.unsqueeze(0))
        return outputs.detach().item()


    def result(self , df ) :

        scale = self.gbp_scale()
        x = self.pros_test(df , scale)
        # model = gbp_lstm()

        return self.pred(model , x)
    














