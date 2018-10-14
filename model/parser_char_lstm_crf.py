import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import *

from module import *


class Parser_Char_LSTM_CRF(torch.nn.Module):
    def __init__(self, parser_layers, parser_dim, n_char, char_dim, char_hidden, n_word, word_dim,
                 n_layers, word_hidden, n_target, drop=0.5):
        super(Parser_Char_LSTM_CRF, self).__init__()

        self.embedding_dim = word_dim
        self.drop1 = torch.nn.Dropout(drop)
        self.embedding = torch.nn.Embedding(n_word, word_dim, padding_idx=0)
        self.scalarmix = ScalarMix(parser_dim, parser_layers, False)
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)

        if n_layers > 1:
            self.lstm_layer = torch.nn.LSTM(
                input_size=self.embedding_dim + char_hidden + parser_dim,
                hidden_size=word_hidden//2,
                batch_first=True,
                bidirectional=True,
                num_layers=n_layers,
                dropout=0.2
            )
        else:
            self.lstm_layer = torch.nn.LSTM(
                input_size=self.embedding_dim + char_hidden + parser_dim,
                hidden_size=word_hidden//2,
                batch_first=True,
                bidirectional=True,
                num_layers=1,
            )
        self.hidden = nn.Linear(word_hidden, word_hidden//2, bias=True)
        self.out = torch.nn.Linear(word_hidden//2, n_target, bias=True)
        self.crf = CRFlayer(n_target)

        self.reset_parameters()

    def load_pretrained_embedding(self, pre_embeddings):
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.embedding.weight = nn.Parameter(pre_embeddings)

    def reset_parameters(self):
        init.xavier_uniform_(self.out.weight)
        init.xavier_uniform_(self.hidden.weight)
        bias = (3.0 / self.embedding.weight.size(1)) ** 0.5
        init.uniform_(self.embedding.weight, -bias, bias)

    def forward(self, word_idxs, char_idxs, parser):
        # mask = torch.arange(x.size()[1]) < lens.unsqueeze(-1)
        mask = word_idxs.gt(0)
        sen_lens = mask.sum(1)

        parser_feature = torch.split(parser, 1, dim=2)
        parser_feature = self.scalarmix(parser_feature)

        char_vec = self.char_lstm.forward(char_idxs[mask])
        char_vec = pad_sequence(torch.split(char_vec, sen_lens.tolist()), True, padding_value=0)

        word_vec = self.embedding(word_idxs)
        feature = self.drop1(torch.cat((word_vec, char_vec, parser_feature), -1))

        sorted_lens, sorted_idx = torch.sort(sen_lens, dim=0, descending=True)
        reverse_idx = torch.sort(sorted_idx, dim=0)[1]
        feature = feature[sorted_idx]
        feature = pack_padded_sequence(feature, sorted_lens, batch_first=True)

        r_out, state = self.lstm_layer(feature, None)
        out, _ = pad_packed_sequence(r_out, batch_first=True, padding_value=0)
        out = out[reverse_idx]
        out = torch.tanh(self.hidden(out))
        out = self.out(out)
        return out

    def forward_batch(self, batch):
        word_idxs, char_idxs, parsers, label_idxs = batch
        mask = word_idxs.gt(0)
        out = self.forward(word_idxs, char_idxs, parsers)
        return mask, out, label_idxs

    def get_loss(self, emit, labels, mask):
        logZ = self.crf.get_logZ(emit, mask)
        scores = self.crf.score(emit, labels, mask)
        return (logZ - scores) / emit.size()[1]
