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
# プログラムについて / How to use

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
Learning will automatically select the GPU if the GPU is available.（Display `CUDA is available:, True`）

そうでなければCPUで学習を実行します。詳しくはPytorchのドキュメントを参照してください。<br>
Otherwise, perform the learning on the CPU. See the Pytorch documentation for details.

GPUの使用状況はコマンドプロンプト及びターミナルから`nvidia-smi -l`で確認します。<br>
GPU usage can be checked from the command prompt and terminal with `nvidia-smi -l`.


GPU利用時、15000データによる学習でバッチサイズ50、Epoch数100でおよそ4時間ほどで計算は終了します。
実際にはEpoch数30以内でだいたい収束します。

- 保存されるモデルについて

モデルは自動で作られる`/model`ディレクトリ下にPTファイルで保存されます。モデルサイズはだいたい６MB程度が予想されます。
保存されるモデルは検証時にもっとも高精度なモデルであると予測される、`model_layer3_2020~日付~.pt`モデルと、
10エポックごとに自動保存される`model_layer3_2020~日付~_Epoch20.pt`があります。
これは過学習を防止するための途中離脱モードを積んでいないためこのような保存方法をとっています。
使用しないモデルは削除しても構いません。

- 学習終了後について

学習終了後、GPUはカーネルの再起動などによって各自で解放してください。

## step2.ipynb
実際に学習済みモデルを使用して音声強調を行います。
クイックな音声の確認を行うためJupyter notebookで書かれています。

`model_path`で使用したいモデルを指定します。
何度もデータセットのローディングを行うことを避けるため、作成したいろんなモデルの聞き比べを行いたい場合
後半のみを実行してください。


# 性能についてのメモ
## 実験１
音声データCMU Arctic コーパス
雑音データUrban Sound 8K

- 学習率0.002、Epoch=20程度でそこそこの精度が出ます。
- ですが、さすがに背景ノイズが大きすぎると強調しきれないようです。
