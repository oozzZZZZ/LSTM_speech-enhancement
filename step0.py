# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 18:51:05 2020

@author: yamamoto
"""

"""
Step 0
学習前のデータ処理フェーズ
"""

import os
import random
import torch

import step_utils as ut

#########################################
###           parameter               ###
#########################################

audio_len = 2**15 #とりあえず従来通り２秒で
# audio_len = 2**9 #512フレームでだいたい３ミリ秒
sample_rate = 16000

# datasets
clean_speech_dir = "./datasets/CMU_ARCTIC"
noise_dir = "./datasets/UrbanSound8K"

datasets_save_dir = "./datasets"


def main():
    print("#####################################################################")
    print("Step0 Make Datasets Phase")
    print("#####################################################################")
    
    if not os.path.exists(datasets_save_dir):
        os.mkdir(datasets_save_dir)
    
    c_files = ut.take_path(clean_speech_dir)
    n_files = ut.take_path(noise_dir)
        
    random.shuffle(c_files)
    random.shuffle(n_files)
    
    n = len(c_files)
    
    a = round(n, -2)
    if a > n:  
        a = round(n-100, -2)
        
    usedata_num = a
    
    print("\nclean speech data is {} files \nNoise data is {} files  \nUse data = {}"
          .format(len(c_files),len(n_files),usedata_num))
    
    c_files,n_files = c_files[:usedata_num],n_files[:usedata_num]
    
    tensor_speech,tensor_addnoise = ut.make_stack(c_files,n_files,audio_len,sample_rate)
    print(tensor_speech.shape,tensor_addnoise.shape)
    torch.save(tensor_speech,datasets_save_dir+"/tensor_speech")
    torch.save(tensor_addnoise,datasets_save_dir+"/tensor_addnoise")
    
    print("#####################################################################")
    print("Step0 All completed!!!")
    print("#####################################################################")


if __name__ == '__main__':
    main()