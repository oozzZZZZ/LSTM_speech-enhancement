# LSTM_speech-enhancement

音声強調モデル作成のためのPytorchプログラムです。<br>
The Pytorch program for speech enhancement.

# 動作環境 / Operating Environment
Windows/Mac/Linuxいずれの環境でも動作確認済みです。<br>
It has been tested in both Windows/Mac/Linux environments.

```
Python 3.7.9

torch ver 1.7.0
tqdm ver 5.0.5
librosa ver 0.8.0

GPU NVIDIA RTX2070 super
CUDA ver11.0
```
**Note** If you have an older version of Pytorch, you will get an error message like the following.
'stft() got an unexpected keyword argument return_complex'
I am currently in the process of fixing this problem.

# プログラムについて / How to use

1. parameter.py : パラメーターの設定(一度だけ実行)
1. step0.py : データセットの前処理を行います。(一度だけ実行)
1. step1.py : 学習を行う
1. step2.ipynb : テストデータを使って音声強調を行います。
1. step3.ipynb : 任意のwavファイルを使って音声強調を行います。

1. parameter.py : Setting parameters (Run only once)
1. step0.py : Processing data sets. (Run only once)
1. step1.py : Training
1. step2.ipynb : Speech enhancement in test data and trained model.
1. step3.ipynb : Speech enhancement in any wav file and trained model.

## parameter.py

データセットのパス指定やモデルのパラメーターまで、学習関連で人間が触っていいものは基本ここだけです。<br>
The parameters operate from here.

```
target_path:　Audio directory path for the audio you want to emphasize.
noise_path:　The directory path of the noise audio you want to remove.
->There is no specification of directory structure since it recursively gets the audio under the specified directory.

datasets_path:　Stores the tensor datasets created in step0.ipynb.
model_path: Save the trained model.
-> datasets_path and model_path will be created automatically if they do not exist.

audio_len: Specify the length of the audio used for learning. Unspecified audio will be deleted.
sample_rate: Sampling frequency. To the specified frequency the audio will be resampled.

noise_rate: You can adjust the magnitude of the noise added to the audio. 0.0 ~ 0.1

datasets_split: test/val/train　Total to be 1.0
batch_size: batch size
num_layer: Specify the number of LSTM layers.

learning_late: learning late

```

強調したい音声のみのwavファイルを格納したディレクトリパスを`clean_speech_dir`に、
除去したい環境音や音楽などのwavファイルを格納したディレクトリパスを`clean_speech_dir`に指定してください。
wavファイルは再帰的に探索されるので、多階層なディレクトリ構造になってても大丈夫です。

この音声ファイルから学習に必要なデータに変換され、作成された`/datasets`ディレクトリに格納されます。
そこそこ大きなデータサイズになります。CMU Arctic CorpusとUrbanSound8k全データ使って合計14GBくらいのデータが作られます。

`num_layer`はLSTMモデルの階層サイズを設定します。値を大きくするほどモデルサイズと学習に使用されるGPUメモリサイズが大きくなります。その代わり精度が上がります。


## step0.py
学習に先立ちデータセットの前処理を行います。<br>
Preprocess the dataset prior to training.

**Note** : quite large data are produced.

## step1.py
学習を行います。<br>
Training Phase

学習はGPUが利用できる環境であれば自動でGPUが選択されます。（`CUDA is available:, True`と表示されます）<br>
そうでなければCPUで学習を実行します。詳しくはPytorchのドキュメントを参照してください。<br>
GPUの使用状況はコマンドプロンプト及びターミナルから`nvidia-smi -l`で確認します。<br>
GPU利用時、15000データによる学習でバッチサイズ50、Epoch数100でおよそ4時間ほどで計算は終了します。<br>
実際にはEpoch数30以内でだいたい収束します。<br>

Learning will automatically select the GPU if the GPU is available.（Display `CUDA is available:, True`）<br>
Otherwise, perform the learning on the CPU. See the Pytorch documentation for details.<br>
GPU usage can be checked from the command prompt and terminal with `nvidia-smi -l`.<br>

Under the following conditions, the calculation time takes about 4 hours.
`15000 Speech / mini batch size = 20 / Epoch = 100`
However, learning converges at about 30 epochs, so there is no need to do even 100 epochs of learning.

- 保存されるモデルについて About saved model data

モデルは自動で作られる`/model`ディレクトリ下にPTファイルで保存されます。モデルサイズはだいたい６MB程度が予想されます。<br>
保存されるモデルは検証時にもっとも高精度なモデルであると予測される、`model_layer3_2020~日付~.pt`モデルと、<br>
10エポックごとに自動保存される`model_layer3_2020~日付~_Epoch20.pt`があります。<br>
これは過学習を防止するための途中離脱モードを積んでいないためこのような保存方法をとっています。<br>
使用しないモデルは削除しても構いません。<br>

The model is saved as a PT file under the automatically created `/model` directory. The model size is expected to be about 6MB.<br>
The models saved are the `model_layer3_2020~Date~.pt` model, which is predicted to be the most accurate model at the time of validation, <br>
and the `model_layer3_2020~Date~_Epoch20.pt` model, which is automatically saved every 10 epochs.<br>
This is because it does not implement a mid-term termination feature to prevent over-learning.<br>
Models that are not used may be removed.

- 学習終了後について After finishing the training

学習終了後、GPUはカーネルの再起動などによって各自で解放してください。<br>
After finishing the training, the GPU should be released on its own, for example by rebooting the kernel.

## step2.ipynb
実際に学習済みモデルを使用して音声強調を行います。<br>
クイックな音声の確認を行うためJupyter notebookで書かれています。

Actual speech enhancement using the trained model.<br>
Written in Jupyter notebook for quick audio confirmation.

Specify the model to be used with `model_path`.<br>
Do not use numbers other than the name of the model file, 
as the program gets the numbers needed for the model from the path `layer3`, etc. 
I am currently working on a fix for this problem.

## step3.ipynb
任意のwavファイルから音声強調を行います。<br>
Perform audio enhancement from any wav file.<br>
Currently supports audio with a length of up to 20 seconds. Any more data will be deleted.

# Datasets used for the experiment
```
Speech data : CMU Arctic
Noise data : Urban Sound 8K
```
