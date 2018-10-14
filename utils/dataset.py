import torch
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence


class TensorDataSet(Data.Dataset):
    def __init__(self, *data):
        super(TensorDataSet, self).__init__()
        self.items = list(zip(*data))

    def __getitem__(self, index):
        return self.items[index]
    
    def __len__(self):
        return len(self.items)


def collate_fn(data):
    batch = zip(*data)
    return tuple([torch.tensor(x) if len(x[0].size()) < 1 else pad_sequence(x, True) for x in batch])


def collate_fn_cuda(data):
    batch = zip(*data)
    return tuple([torch.tensor(x).cuda() if len(x[0].size()) < 1 else pad_sequence(x, True).cuda() for x in batch])
