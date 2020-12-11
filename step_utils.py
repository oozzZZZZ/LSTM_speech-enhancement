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
import parameter
from scipy.signal import fftconvolve
p=parameter.Parameter()

def take_path(path):
    data_list = []
    for a,b,c in os.walk(path):
        l = glob.glob(a+"/*.wav" ,recursive=True)
        data_list = data_list + l
    return data_list

# data augumentation tools

def time_shift(data,sample_rate,shift):
    data_roll = np.roll(data, shift)
    return data_roll

def stretch(data, rate=1):
    input_length = len(data)
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data

def pitch_shift(data,sample_rate,shift):
    """
    shiftについて
    +12:1オクターブ上
    -12:1オクターブ下
    +7:完全５度
    +5:完全４度
    """
    ret = librosa.effects.pitch_shift(data, sample_rate, shift, bins_per_octave=12, res_type='kaiser_best')
    return ret

def mk_reverb_ir(ir_len=1, rt=1, fs=16000, init_rev=True):
    
    t = np.linspace(0, ir_len, ir_len*fs)
    E = np.power(10, -3 * t / rt)
    
    reverb = np.power(E, 2/3)
    rand = np.random.randint(0, 2, reverb.size)
    rand = np.where(rand==1, 1, -1)
    reverb *= rand
    
    if init_rev:
        t = np.linspace(0, 0.1, int(fs*0.1))
        rand_t = np.random.rand(t.size)
        density = 8 * t + 0.2
        rand_density = density - rand_t
        rand_init_reverb = np.where(rand_density>0, 1, 0)
        reverb[:int(0.1*fs)] *= rand_init_reverb
    
    return reverb

def conv_reverb(data,sample_rate,rt):
    
    reverb = mk_reverb_ir(ir_len=rt, rt=rt, fs=sample_rate)
    data_reverb = fftconvolve(data, reverb)
    
    return data_reverb

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

def dataloading_skip(data,audio_len): # 音声データが破損してそうな場合スキップしたい
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

def make_stack(c_files,n_files,audio_len,sample_rate,noise_snr,augmentation_mode):
    speech_list = []
    speech_noise_list = []
    
    num_c_files = len(c_files)
    num_n_files = len(n_files)
    
    print("\nclean speech data is {} files \nNoise data is {} files"
          .format(num_c_files,num_n_files))        
    
    for c in tqdm(c_files):
        n = n_files[random.randint(0, num_n_files-1)]
        
        
        # 音声データが破損してそうな場合スキップしたい
        c_data, _  = librosa.load(c, sr=sample_rate)
        n_data, _  = librosa.load(n, sr=sample_rate)
        
        skip_count = 0
        
        #破損データのスキップ
        if len(c_data) > audio_len and len(n_data) > 10:
            
            
            #データ水増しモードオン
            if augmentation_mode == True:
                step = len(c_data) // audio_len
                for i in range(step):
                    for w in range(p.augmentation_num):
                        c_p = c_data[i*audio_len : (i+1)*audio_len]
                        c_p = stretch(c_p, rate=random.uniform(0.5, 1.5)) #ランダムデータ伸縮
                        c_p = pitch_shift(c_p,16000,random.uniform(-12, 12)) #ランダムピッチシフト
                        c_p = conv_reverb(c_p, 16000, random.uniform(0, 1.0))
                        c_p = c_p * random.uniform(0.8, 1.2) #ランダム音量変更
                        c_stft = dataloading_skip(c_p,audio_len)
                        n_stft = dataloading_skip(n_data,audio_len)
                        c_n_stft = addnoise(c_stft,n_stft,noise_snr)
                        speech_list.append(c_stft)
                        speech_noise_list.append(c_n_stft)
                    
            #データ水増しモードオフ    
            else:
                step = len(c_data) // audio_len
                for i in range(step):
                    c_p = c_data[i*audio_len : (i+1)*audio_len]
                    c_stft = dataloading_skip(c_p,audio_len)
                    n_stft = dataloading_skip(n_data,audio_len)
                    c_n_stft = addnoise(c_stft,n_stft,noise_snr)
                    speech_list.append(c_stft)
                    speech_noise_list.append(c_n_stft)

        else:
            skip_count += 1
    
    #データの出力処理
        
    num_data = len(speech_list)
    a = round(num_data, -2)
    if a > num_data:  
        usedata_num = round(num_data-100, -2)

    tensor_speech = torch.stack(speech_list[:usedata_num])
    tensor_addnoise = torch.stack(speech_noise_list[:usedata_num])
    
    print("Maybe Corrupt Data :",skip_count)
    print("Available data :", num_data)
    print("Use data :", usedata_num)
    
    return tensor_speech,tensor_addnoise
