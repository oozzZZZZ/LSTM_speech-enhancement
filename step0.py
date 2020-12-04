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
import parameter


def main():
    print("#####################################################################")
    print("Step0 Make Datasets Phase")
    print("#####################################################################")
    
    p=parameter.Parameter()
    audio_len = p.audio_len
    sample_rate = p.sample_rate
    noise_snr = p.noise_rate 
    clean_speech_dir = p.target_path
    noise_dir = p.noise_path
    datasets_save_dir = p.datasets_path
    augmentation_mode = p.augmentation_mode
    
    if not os.path.exists(datasets_save_dir):
        os.mkdir(datasets_save_dir)
    
    c_files = ut.take_path(clean_speech_dir)
    n_files = ut.take_path(noise_dir)
        
    random.shuffle(c_files)
    random.shuffle(n_files)
    
    tensor_speech,tensor_addnoise = ut.make_stack(c_files,n_files,audio_len,sample_rate,noise_snr,augmentation_mode)
    print(tensor_speech.shape,tensor_addnoise.shape)
    
    torch.save(tensor_speech,datasets_save_dir+"/tensor_speech")
    torch.save(tensor_addnoise,datasets_save_dir+"/tensor_addnoise")
    
    print("#####################################################################")
    print("Step0 All completed!!!")
    print("#####################################################################")


if __name__ == '__main__':
    main()