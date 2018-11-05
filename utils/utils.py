import pickle

import h5py
import torch


def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def read_elmo(file, num_representation):
    assert (0<num_representation<=3)
    result = []
    h = h5py.File(file, 'r')
    sen_num = len(h.keys())-1
    result = [torch.tensor(h.get(str(i)))[0:num_representation].transpose(
        0, 1) for i in range(sen_num)]
    return result
