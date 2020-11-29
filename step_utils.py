# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 18:46:45 2020

@author: yamamoto
"""
import torch
import librosa
import numpy as np
from tqdm import tqdm
import random
import os
import glob


def length_fitting(data,audio_len):
    if len(data) > audio_len:
        data = data[:audio_len]
    else: 
        while len(data) < audio_len:
            data = torch.cat((data,data),0)[:audio_len]
    return data
    
def wav2tensorstft(filename,audio_len,sample_rate):
    data, _  = librosa.load(filename, sr=sample_rate)
    data=torch.from_numpy(data.astype(np.float32)).clone()
    data = length_fitting(data,audio_len)
    stftdata = torch.stft(data,2**8,return_complex=True)
    realpart = stftdata.real
    imagpart = stftdata.imag
    c = torch.cat((realpart,imagpart),0)
    return c

def tensorstft2audio(stftdata):
    h = int(stftdata.shape[0]/2)
    z = torch.complex(stftdata[:h,:], stftdata[h:,:])
    tensoraudio = torch.istft(z,2**8)
    numpyaudio = tensoraudio.to('cpu').detach().numpy().copy()
    return numpyaudio
    
def addnoise(speech_stft,noise_stft,noise_snr):
    nsr_noise_stft = noise_stft * noise_snr
    addnoise = torch.add(speech_stft,nsr_noise_stft)
    return addnoise

def make_stack(c_files,n_files,audio_len,sample_rate,noise_snr):
    speech_list = []
    speech_noise_list = []
    num_n_files = len(n_files)
    for c in tqdm(c_files):
        n = n_files[random.randint(0, num_n_files-1)]
        c_stft = wav2tensorstft(c,audio_len,sample_rate)
        n_stft = wav2tensorstft(n,audio_len,sample_rate)
        c_n_stft = addnoise(c_stft,n_stft,noise_snr)
        speech_list.append(c_stft)
        speech_noise_list.append(c_n_stft)

    tensor_speech = torch.stack(speech_list)
    tensor_addnoise = torch.stack(speech_noise_list)
    return tensor_speech,tensor_addnoise

def take_path(path):
    data_list = []
    for a,b,c in os.walk(path):
        l = glob.glob(a+"/*.wav" ,recursive=True)
        data_list = data_list + l
    return data_list