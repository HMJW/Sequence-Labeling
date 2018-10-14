import collections
import pickle
from itertools import chain

from utils.vocab import Vocab


def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def collect(corpus, low_freq=1):
    labels = sorted(set(chain(*corpus.label_seqs)))
    words = collections.Counter(chain(*corpus.word_seqs))
    words = [w for w,f in words.items() if f > low_freq]
    chars = sorted(set(''.join(words)))
    vocab = Vocab(words, labels, chars)
    return vocab
