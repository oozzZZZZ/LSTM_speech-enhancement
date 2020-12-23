#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:31:07 2020
@author: t.yamamoto
"""

"""
Step 1
学習フェーズ
"""
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.optim as optim

import os
import numpy as np
import datetime
import glob
from tqdm import tqdm

import model as mm
import parameter

def main():

    p=parameter.Parameter()
    
    datasets_save_dir = p.datasets_path
    model_save_dir = p.model_path
    split = p.datasets_split
    batch_size = p.batch_size
    learning_late = p.learning_late
    num_layer = p.num_layer
    epochs = p.epochs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    print("CUDA is available:", torch.cuda.is_available())
    
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
        
    now = datetime.datetime.now()
    model_path = model_save_dir+"/model_layer"+str(num_layer)+"_"+now.strftime('%Y%m%d_%H%M%S')+".pt"
    
    _datasets_path = glob.glob(datasets_save_dir+"/*.npz")
    
    speech_list = []
    addnoise_list = []
    
    print("load npz data and transform it stacked tensor...")
    
    for file in tqdm(_datasets_path):
        d = np.load(file)    
        speech=torch.from_numpy(d["speech"].astype(np.float32)).clone()
        addnoise=torch.from_numpy(d["addnoise"].astype(np.float32)).clone()
        
        speech_list.append(speech)
        addnoise_list.append(addnoise)
        
    num_data = len(speech_list)
    a = round(num_data, -2)
    if a > num_data:  
        num_usedata = round(num_data-100, -2)
        
    tensor_speech = torch.stack(speech_list[:num_usedata])
    tensor_addnoise = torch.stack(addnoise_list[:num_usedata])
    
    print("Available data :", num_data)
    print("Use data :", num_usedata)
    
    mydataset = utils.TensorDataset(tensor_speech,tensor_addnoise)
    data_num = tensor_speech.shape[0]
    data_split = [int(data_num * split[0]),
                  int(data_num * split[1]),
                  int(data_num * split[2])]
    test_dataset,val_dataset,train_dataset = utils.random_split(mydataset,data_split)
    
    train_loader = utils.DataLoader(train_dataset,batch_size=batch_size,num_workers=os.cpu_count(),pin_memory=True,shuffle=True)
    val_loader = utils.DataLoader(val_dataset,batch_size=batch_size,num_workers=os.cpu_count(),pin_memory=True,shuffle=True)
    test_loader = utils.DataLoader(test_dataset,batch_size=batch_size,num_workers=os.cpu_count(),pin_memory=True,shuffle=True)
    
    # model
    feat = tensor_addnoise.shape[1]
    sequence = tensor_addnoise.shape[2]
    model = mm.Net(sequence, feat, num_layer).to(device)
    
    #loss/optimizer   
    criterion = nn.L1Loss().to(device)
    # criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_late)
    
    print("#####################################################################")
    print(" Start Training..")
    print("#####################################################################")
    
    
    train_loss_list = []
    test_loss_list = []
    
    for epoch in tqdm(range(1, epochs+1),desc='[Training..]'):
    
    
        # Training
    
        model.train()  # 訓練モードに
    
        train_loss = 0
        for batch_idx, (speech, addnoise) in enumerate(train_loader):
            # データ取り出し
            speech, addnoise = speech.to(device), addnoise.to(device)
            optimizer.zero_grad()
            # 伝搬
    
            mask = model(addnoise) #modelでmask自体を推定する
            h_hat = mask * addnoise #雑音つき音声にmaskを適応し所望音声を強調
    
            # 損失計算とバックプロパゲーション
            loss = criterion(h_hat, speech) #強調音声 vs ラベル
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)
    
        # Eval
    
        model.eval()
    
        test_loss = 0
        with torch.no_grad():
            for speech, addnoise in val_loader:
                # データ取り出し
                speech, addnoise = speech.to(device), addnoise.to(device)
                mask = model(addnoise)
                h_hat = mask * addnoise 
                test_loss += criterion(h_hat, speech).item()  # sum up batch loss
    
            test_loss /= len(test_loader.dataset)
            test_loss_list.append(test_loss)
    
            tqdm.write('\nTrain set: Average loss: {:.6f}\nTest set:  Average loss: {:.6f}'
                       .format(train_loss,test_loss))
    
        if epoch == 1:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)
    
        else:
            if best_loss > test_loss:
                torch.save(model.state_dict(), model_path)
                best_loss = test_loss
    
    
        if epoch % 10 == 0: #10回に１回定期保存
            epoch_model_path = model_save_dir+"/model_layer"+str(num_layer)+"_"+now.strftime('%Y%m%d_%H%M%S')+"_Epoch"+str(epoch)+".pt"
            torch.save(model.state_dict(), epoch_model_path)
            
if __name__ == '__main__':
    main()
