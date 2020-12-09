# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 18:51:05 2020

@author: yamamoto

パラメーター関連
"""

class Parameter():
    def __init__(self):
        """
        data path
        '''
        target_path:強調したい音声を入れてください
        noise_path:除去したい背景雑音を入れてください
        ->指定されたディレクトリ下の音声を再起的に取得するのでディレクトリ構造の指定は無し
        
        datasets_path:step0で作成されるtensorデータセットを格納します。
        model_path:学習されたモデルを保存します
        ->ディレクトリがない場合自動生成されます。
        """
        self.target_path = "./datasets/CMU_ARCTIC"
        self.noise_path = "./datasets/UrbanSound8K"

        self.datasets_path = "./datasets"

        self.model_path = "./model/"
        
        """
        音声データに関するパラメータ
        """
        self.audio_len = 2**15 #2**15=2s,2**9=３ms
        self.sample_rate = 16000
        
        self.noise_rate = 0.7 #付加されるノイズの大きさ(0.0~1.0)
        
        """
        学習に関するパラメータ
        """
        
        self.datasets_split = [0.1,0.2,0.7] #test/val/train
        self.batch_size = 50
        self.num_layer = 5 #LSTMレイヤー数
        self.learning_late = 0.002
        
        #学習時のデータ水増し処理を行うか
        self.augmentation_mode = True
        self.augmentation_num = 4 #何倍に増やす？
    
if __name__ == '__main__':
    p = Parameter()
    print(p.sample_rate)
