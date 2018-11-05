# 	POS Tagging for Chinese

Model : CharLSTM + LSTM + CRF
​              Elmo + LSTM + CRF
​              Parser + CharLSTM + LSTM + CRF

Pretrained Word Embedding : glove.6B.100d.txt

NER Data : CoNLL03
Chunking Data : CoNLL00
POS Data : WSJ

Other resources:
Elmo : produce elmo representation on all datasets, and save on the disk
Biaffine Parser : produce biaffine parser's LSTMs output on all datasets, and save on the disk

## requirements

```
python >= 3.6.3
pytorch = 0.4.1
```

## running

```
mkdir save                                     # or define other path to save models and vocabs
python train.py --pre_emb --task=ner --gpu=0   # choose task and if use gpu and pretrain embedding
python train_parser --pre_emb --task=ner --gpu=0
python train_elmo.py --pre_emb --task=
```

## results

##### Task:NER

##### Data:CoNLL03

##### Pretrained Embedding:[glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/).

| model                    | dev    | test   | Iter  |
| ------------------------ | ------ | ------ | ----- |
| CharLSTM+LSTM+CRF        | 94.47% | 91.10% | 18/29 |
| Elmo+LSTM+CRF            | 95.56% | 92.12% | 25/36 |
| Parser+CharLSTM+LSTM+CRF | 94.87% | 90.87% | 35/46 |



##### Task:Chunking

##### Data:CoNLL00

##### Pretrained Embedding:[glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/).

| model                    | dev    | test   | Iter  |
| ------------------------ | ------ | ------ | ----- |
| CharLSTM+LSTM+CRF        | 94.99% | 94.38% | 18/29 |
| Elmo+LSTM+CRF            | 97.13% | 96.37% | 23/34 |
| Parser+CharLSTM+LSTM+CRF | 96.41% | 96.48% | 10/21 |
| Parser+Elmo+LSTM+CRF     | 96.98% | 96.59% | 12/23 |



##### Task:POS

##### Data:WSJ

##### Pretrained Embedding:[glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/).

| model                    | dev    | test   | Iter  |
| ------------------------ | ------ | ------ | ----- |
| CharLSTM+LSTM+CRF        | 97.67% | 97.67% | 10/21 |
| Elmo+LSTM+CRF            | 97.89% | 97.78% | 3/14  |
| Parser+ChatLSTM+LSTM+CRF | 97.89% | 97.70% | 6/17  |
| Parser+Elmo+LSTM+CRF     | 97.92% | 97.78% | 2/13  |

​	