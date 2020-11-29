# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:48:46 2020

@author: yamamoto
"""
import torch.nn as nn


class Net(nn.Module):
        def __init__(self,sequence_length, feat_size, num_layer):
            super(Net, self).__init__()
            self.seq_len = sequence_length
            self.feature_size = feat_size
            self.hidden_layer_size = feat_size #隠れ層サイズ
            self.rnn_layers = num_layer #RNNのレイヤー層
            self.lstm=nn.LSTM(input_size = self.feature_size,
                     hidden_size = self.hidden_layer_size,
                     num_layers = self.rnn_layers,
                     dropout = 0.05,
                     batch_first = True)
            
        def forward(self,x):
            #[Batch, feature, sequence] -> [Batch, sequence, feature]
            x = x.permute(0,2,1)
            rnn_out,(h_n,c_n)= self.lstm(x)
            #[Batch, sequence, feature] -> [Batch, feature, sequence]
            y = rnn_out.permute(0,2,1)
            return y