# An RNN-based Seq2Seq model
## Installation
**Python versioin** 

```
Python 3.6.2
```

**Package Requirements** 

```
Torch 1.0.1.post2, Numpy 1.15.1, Tqdm 4.31.1, Rouge 0.3.2, Scipy 1.1.0
```

## Date Processing
```
python build_data.py
```

## Train model
```
python train.py -batch 256 -epoch 20 -n_layers 2 -save_model
```

Some model parameters can be set in the utils/config.py

* self.cell: Setting rnn unit is \'LSTM\' or \'GRU\'.

* self.bidirectional: Whether the setting unit is bidirectional or not. 

* self.attn_flag: Different Calculating Functions of Attention Mechanism

  * loung: [Loung attention](https://arxiv.org/pdf/1409.0473.pdf)
                 
  * bahdanau: [Bahdanau attention](https://arxiv.org/pdf/1508.04025.pdf)
                 
* self.enc_attn: Whether or not to apply self-attention mexhanism at encoder

* self.intra_decoder: Whether or not to apply self-attention mexhanism at decoder

* self.cnn:

  * cnn=0: No feature extraction of sentences

  * cnn=1: Using CNN to extract sentence features and connect the feature vectors to the final hidden state of the encoder.
           
  * cnn=2: Using CNN to extract sentence features and connect the feature vectors to the encoder outputs.
           
## Test

```
python beam_test.py -b 5
```

**PS** Setting up the file of the model.

## Result
|  Model | R-1 | R-2 | R-L | 
| - | -: | -: | -: | 
| Bi-GRU | 0.2917 | 0.1244	| 0.2347 |
| Bi-GRU + Bahdanau | 0.3035 |	0.1411 | 0.2547 |
| Bi-LSTM + Bahdanau | 0.3623 | 0.2198 | 0.3231 |
| Bi-LSTM + Luong | 0.3621	| 0.2321 | 0.3301 |
| Bi-LSTM + Luong + enc_attn | 0.3920	| 0.2615 |	0.3452 |
| Bi-LSTM + Luong + inter_dec |  0.3847 | 0.2471	| 0.3402 |
| Bi-LSTM + Luong + CNN(1) |  0.3881 | 0.2602 | 0.3586 |
| Bi-LSTM + Luong + CNN(2) | 0.3914 | 0.2645 | 0.3701 |

