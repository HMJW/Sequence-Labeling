import collections
import pickle
from itertools import chain

import torch
import torch.nn.init as init


class Vocab(object):
    def __init__(self, corpus, lower=False, min_freq=1):
        words, chars, labels = self.collect(corpus, min_freq)

        #  if lower=True,lower all the words.But not all Chars
        if lower:
            words = [w.lower() for w in words]
        #  ensure the <PAD> index is 0
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'

        self._words = [self.PAD] + words + [self.UNK]
        self._chars = [self.PAD] + chars + [self.UNK]
        self._labels = labels

        self._word2id = {w: i for i, w in enumerate(self._words)}
        self._char2id = {c: i for i, c in enumerate(self._chars)}
        self._label2id = {l: i for i, l in enumerate(self._labels)}
        
    @staticmethod
    def collect(corpus, min_freq=1):
        labels = sorted(set(chain(*corpus.label_seqs)))
        words = sorted(chain(*corpus.word_seqs))
        
        chars_freq = collections.Counter(''.join(words))
        chars = [c for c,f in chars_freq.items() if f >= min_freq]
        words_freq = collections.Counter(words)
        words = [w for w,f in words_freq.items() if f>= min_freq]
       
        return words, chars, labels

    def read_embedding(self, embedding_file, unk_in_pretrain='unk'):
        #  ensure the <PAD> index is 0
        with open(embedding_file, 'r') as f:
            lines = f.readlines()
        splits = [line.split() for line in lines]
        # read pretrained embedding file
        words, vectors = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])

        if isinstance(unk_in_pretrain, str):
            assert unk_in_pretrain in words
            words = list(words)
            words[words.index(unk_in_pretrain)] = self.UNK

        pretrained = {w: torch.tensor(v) for w, v in zip(words, vectors)}
        out_train_words = [w for w in words if w not in self._word2id]
        out_train_chars = [c for c in ''.join(out_train_words) if c not in self._char2id]

        # extend words and chars
        # ensure the <PAD> token at the first position
        self._words =[self.PAD] + sorted(set(self._words + out_train_words) - {self.PAD})
        self._chars =[self.PAD] + sorted(set(self._chars + out_train_chars) - {self.PAD})

        # update the words,chars dictionary
        self._word2id = {w: i for i, w in enumerate(self._words)}
        self._char2id = {c: i for i, c in enumerate(self._chars)}

        # initial the extended embedding table
        embdim = len(vectors[0])
        extended_embed = torch.randn(self.num_words, embdim)
        bias = (3.0 / embdim) ** 0.5
        init.uniform_(extended_embed, -bias, bias)
        
        # the word in pretrained file use pretrained vector
        # the word not in pretrained file but in training data use random initialized vector
        for i, w in enumerate(self._words):
            if w in pretrained:
                extended_embed[i] = pretrained[w]
            elif w.lower() in pretrained:
                extended_embed[i] = pretrained[w.lower()]
        return extended_embed

    @property
    def num_words(self):
        return len(self._words)

    @property
    def num_labels(self):
        return len(self._labels)

    @property
    def num_chars(self):
        return len(self._chars)
    
    @property
    def UNK_word_index(self):
        return self._word2id[self.UNK]

    @property
    def UNK_char_index(self):
        return self._char2id[self.UNK]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def __repr__(self):
        return 'Words : %d，Characters : %d，labels : %d' % (self.num_words, self.num_chars, self.num_labels)

    def word2id(self, word, lower=False):
        def f(x):
            if x in self._word2id:
                return self._word2id[x]
            elif x.lower() in self._word2id:
                return self._word2id[x.lower()]
            else:
                return self._word2id[self.UNK]
        assert (isinstance(word, str) or isinstance(word, list))
        if isinstance(word, str):
            if lower:
                return self._word2id.get(word.lower(), self.UNK_word_index)
            else:
                return f(word)
        elif isinstance(word, list):
            if lower:
                return [self._word2id.get(w.lower(), self.UNK_word_index) for w in word]
            else:
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
