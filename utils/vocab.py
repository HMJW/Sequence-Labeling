import collections
from itertools import chain

import torch
import torch.nn.init as init


class Vocab(object):
    def collect(self, corpus, min_freq=1):
        labels = sorted(set(chain(*corpus.label_seqs)))
        words = collections.Counter(chain(*corpus.word_seqs))
        words = [w for w,f in words.items() if f > min_freq]
        chars = sorted(set(''.join(words)))
        return words, chars, labels

    def __init__(self, corpus, min_freq=1):
        words, chars, labels = self.collect(corpus, min_freq)
        #  ensure the <PAD> index is 0
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'

        self._words = [self.PAD] + words + [self.UNK]
        self._chars = [self.PAD] + chars + [self.UNK]
        self._labels = labels

        self._word2id = {w: i for i, w in enumerate(self._words)}
        self._char2id = {c: i for i, c in enumerate(self._chars)}
        self._label2id = {l: i for i, l in enumerate(self._labels)}

        self.num_words = len(self._words)
        self.num_chars = len(self._chars)
        self.num_labels = len(self._labels)

        self.UNK_word_index = self._word2id[self.UNK]
        self.UNK_char_index = self._char2id[self.UNK]
        self.PAD_word_index = self._word2id[self.PAD]
        self.PAD_char_index = self._char2id[self.PAD]

    def read_embedding(self, embedding_file):
        'ensure the <PAD> index is 0'
        with open(embedding_file, 'r') as f:
            lines = f.readlines()
        splits = [line.split() for line in lines]
        # read pretrained embedding file
        words, vectors = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])

        pretrained = {w: torch.tensor(v) for w, v in zip(words, vectors)}
        unk_words = [w for w in words if w not in self._word2id]
        unk_chars = [c for c in ''.join(unk_words) if c not in self._char2id]

        # extend words and chars
        # ensure the <PAD> token at the first position
        self._words =[self.PAD] + sorted(set(self._words + unk_words) - {self.PAD})
        self._chars =[self.PAD] + sorted(set(self._chars + unk_chars) - {self.PAD})

        # update the words,chars dictionary
        self._word2id = {w: i for i, w in enumerate(self._words)}
        self._char2id = {c: i for i, c in enumerate(self._chars)}
        self.UNK_word_index = self._word2id[self.UNK]
        self.UNK_char_index = self._char2id[self.UNK]
        self.PAD_word_index = self._word2id[self.PAD]
        self.PAD_char_index = self._char2id[self.PAD]
        
        # update the numbers of words and chars
        self.num_words = len(self._words)
        self.num_chars = len(self._chars)

        # initial the extended embedding table
        embdim = len(vectors[0])
        
        extended_embed = torch.randn(self.num_words, embdim)
        bias = (3.0 / embdim) ** 0.5
        init.uniform_(extended_embed, -bias, bias)

        # different from chinese POS
        # the word in pretrained file use pretrained vector
        # the word not in pretrained file but in training data use random initialized vector
        for i, w in enumerate(self._words):
            if w in pretrained:
                extended_embed[i] = pretrained[w]
            elif w.lower() in pretrained:
                extended_embed[i] = pretrained[w.lower()]
        return extended_embed

    def word2id(self, word):
        'different from Chinese POS'
        def f(x):
            if x in self._word2id:
                return self._word2id[x]
            elif x.lower() in self._word2id:
                return self._word2id[x.lower()]
            else:
                return self._word2id[self.UNK]

        assert (isinstance(word, str) or isinstance(word, list))
        if isinstance(word, str):
            return f(word)
        elif isinstance(word, list):
            return [f(w) for w in word]

    def label2id(self, label):
        assert (isinstance(label, str) or isinstance(label, list))
        if isinstance(label, str):
            return self._label2id.get(label, 0) # if label not in training data, index to 0 ?
        elif isinstance(label, list):
            return [self._label2id.get(l, 0) for l in label]

    def char2id(self, char, max_len=20):
        assert (isinstance(char, str) or isinstance(char, list))
        if isinstance(char, str):
            return self._char2id.get(char, self.UNK_char_index)
        elif isinstance(char, list):
            return [[self._char2id.get(c, self.UNK_char_index) for c in w[:max_len]] + 
                    [0] * (max_len - len(w)) for w in char]

    def id2label(self, id):
        assert (isinstance(id, int) or isinstance(id, list))
        if isinstance(id, int):
            assert (id >= self.num_labels)
            return self._labels[id]
        elif isinstance(id, list):
            return [self._labels[i] for i in id]
