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
| CharLSTM+LSTM+CRF        | 95.06% | 94.53% | 18/29 |
| Elmo+LSTM+CRF            | 97.02% | 96.31% | 37/48 |
| Parser+CharLSTM+LSTM+CRF | 96.36% | 96.28% | 21/10 |
| Parser+Elmo+LSTM+CRF     | 97.08% | 96.53% | 29/40 |



##### Task:POS

##### Data:WSJ

##### Pretrained Embedding:[glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/).

| model                    | dev    | test   | Iter  |
| ------------------------ | ------ | ------ | ----- |
| CharLSTM+LSTM+CRF        | 97.71% | 97.67% | 10/21 |
| Elmo+LSTM+CRF            | 97.88% | 97.76% | 7/18  |
| Parser+ChatLSTM+LSTM+CRF | 97.91% | 97.70% | 6/17  |
| Parser+Elmo+LSTM+CRF     | 97.94% | 97.76% | 2/13  |

​	