# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:25:43 2020

@author: yamamoto
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
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

import model as mm
#########################################
####           Parameter             ####
#########################################
#training parameter
datasets_save_dir = "./datasets"
model_save_dir = './model'
split = [500,2000,12500] #test/val/train
batch_size = 10
epochs = 100
learning_late = 0.002

#model parameter
num_layer = 6

#########################################
####             Main                ####
#########################################
def main():
    print("#####################################################################")
    print("Step1 Training Phase")
    print("#####################################################################")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("CUDA is available:", torch.cuda.is_available())
    print("\nLoading Datasets.....")
    
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
        
    #保存モデル名の定義
    now = datetime.datetime.now()
    model_path = model_save_dir+"/model_layer"+str(num_layer)+"_"+now.strftime('%Y%m%d_%H%M%S')+".pt"
        
    tensor_speech = torch.load(datasets_save_dir+"/tensor_speech")
    tensor_addnoise = torch.load(datasets_save_dir+"/tensor_addnoise")
    
    mydataset = utils.TensorDataset(tensor_speech,tensor_addnoise)
    test_dataset,val_dataset,train_dataset = utils.random_split(mydataset,split)
    
    # Shuffleしない # データローダー：フル
    train_loader = utils.DataLoader(train_dataset,batch_size=batch_size,num_workers=os.cpu_count(),pin_memory=True)
    val_loader = utils.DataLoader(val_dataset,batch_size=batch_size,num_workers=os.cpu_count(),pin_memory=True)
    test_loader = utils.DataLoader(test_dataset,batch_size=batch_size,num_workers=os.cpu_count(),pin_memory=True)
    
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
        

    
    fig,ax = plt.figure()
    ax.plot(train_loss_list,linewidth=2, color="red" ,label="Train Loss")
    ax.plot(test_loss_list,linewidth=2, color="blue" ,label="Test Loss")
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(model_save_dir+"/model_layer"+str(num_layer)+"_"+now.strftime('%Y%m%d_%H%M%S')+".png")


if __name__ == '__main__':
    main()