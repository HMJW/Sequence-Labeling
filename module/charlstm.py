import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import *


class CharLSTM(torch.nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()
        self.char_dim = char_dim
        self.char_embedding = torch.nn.Embedding(n_char, char_dim, padding_idx=0)
        self.char_lstm = torch.nn.LSTM(
            input_size=char_dim,
            hidden_size=char_hidden//2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        bias = (3 / self.char_embedding.weight.size(1)) ** 0.5
        nn.init.uniform_(self.char_embedding.weight, -bias, bias)
        # nn.init.xavier_uniform_(self.char_embedding.weight)

    def forward(self, chars):
        mask = chars.gt(0)
        chars_lens = mask.sum(1)
        
        sorted_lens, sorted_index = torch.sort(chars_lens, dim=0, descending=True)
        maxlen = sorted_lens[0]
        sorted_chars = chars[sorted_index, :maxlen]
        sorted_chars = self.char_embedding(sorted_chars)
        raw_index = torch.sort(sorted_index, dim=0)[1]

        input = pack_padded_sequence(sorted_chars, sorted_lens, batch_first=True)
        out, h = self.char_lstm(input, None)
        hidden_state = torch.cat((h[0][0], h[0][1]), 1)
        return torch.index_select(hidden_state, 0, raw_index)
