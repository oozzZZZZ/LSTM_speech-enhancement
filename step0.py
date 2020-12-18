#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:31:07 2020

@author: t.yamamoto
"""

import numpy as np
import os
import random
from tqdm import tqdm

from librosa.core import load, stft

import utils as ut
import parameter

p=parameter.Parameter()

audio_len = p.audio_len
sample_rate = p.sample_rate
clean_speech_dir = p.target_path
noise_dir = p.noise_path
datasets_save_dir = p.datasets_path

fft_size = p.fft_size
hop_length = p.hop_length

def length_fitting(data,audio_len):
    if len(data) > audio_len:
        data = data[:audio_len]
    else: 
        while len(data) < audio_len:
            data = np.concatenate((data,data),0)[:audio_len]
    return data

def main():

    if not os.path.exists(datasets_save_dir):
        os.mkdir(datasets_save_dir)
        
    c_files = ut.take_path(clean_speech_dir)
    n_files = ut.take_path(noise_dir)
    
    random.shuffle(c_files)
    random.shuffle(n_files)
    
    num_c_files = len(c_files)
    num_n_files = len(n_files)
    
    print("\nclean speech data is {} files \nNoise data is {} files"
              .format(num_c_files,num_n_files)) 
    
    def length_fitting(data,audio_len):
        if len(data) > audio_len:
            data = data[:audio_len]
        else: 
            while len(data) < audio_len:
                data = np.concatenate((data,data),0)[:audio_len]
        return data
    
    data_idx = 0
    
    for c in tqdm(c_files,leave=True):
        data_p_idx = 0
        c_data, sr_c = load(c, sr=None)
        
        if sr_c != sample_rate:
            c_data, _ = load(c, sr=sample_rate)
            
        skip_count = 0
        
        if len(c_data) < audio_len:
            skip_count += 1
            
        else:
            step = len(c_data) // audio_len
            
            for i in tqdm(range(step),leave=False):
                c_p = c_data[i*audio_len : (i+1)*audio_len]
                
                n = n_files[random.randint(0, num_n_files-1)]
                n_data, sr_n = load(n, sr=None)
                if sr_n != sample_rate:
                    n_data, _ = load(n, sr=sample_rate)
                n_data = length_fitting(n_data,audio_len)
                
                
                c_p = np.abs(stft(c_p, n_fft=fft_size, hop_length=hop_length)).astype(np.float32)
                n_data = np.abs(stft(n_data, n_fft=fft_size, hop_length=hop_length)).astype(np.float32)
                c_n_data = c_p + n_data
                norm = c_p.max()
    
                c_p /= norm
                n_data /= norm
                c_n_data /= norm
                
                np.savez(os.path.join(datasets_save_dir, str(data_idx)+"_"+str(data_p_idx)+".npz"),
                         speech=c_p, addnoise=c_n_data)
                
                data_p_idx += 1
    
        data_idx += 1
        
if __name__ == '__main__':
    main()
