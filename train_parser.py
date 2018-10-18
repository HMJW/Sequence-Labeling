import argparse

import torch
import torch.utils.data as Data

from config import config
from model import Parser_Char_LSTM_CRF
from utils import *


def process_data(vocab, dataset, parser, parser_layers=3, lower=False, max_word_len=20):
    assert (0<parser_layers<=3)
    word_idxs, char_idxs, parsers, label_idxs = [], [], [], []

    for wordseq, labelseq, p in zip(dataset.word_seqs, dataset.label_seqs, parser):
        _word_idxs = vocab.word2id(wordseq, lower)
        _label_idxs = vocab.label2id(labelseq)
        _char_idxs = vocab.char2id(wordseq, max_word_len)

        word_idxs.append(torch.tensor(_word_idxs))
        char_idxs.append(torch.tensor(_char_idxs))
        label_idxs.append(torch.tensor(_label_idxs))
        parsers.append(torch.tensor(p, dtype=torch.float)[:,:parser_layers])

    return TensorDataSet(word_idxs, char_idxs, parsers, label_idxs)


if __name__ == '__main__':
    # init config
    model_name = 'parser_char_lstm_crf'
    config = config[model_name]
    for name, value in vars(config).items():
        print('%s = %s' %(name, str(value)))
        
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu', type=int, default=config.gpu, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--pre_emb', action='store_true', help='choose if use pretrain embedding')
    parser.add_argument('--task', choices=['pos', 'chunking', 'ner'], default='chunking', help='task choice')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--thread', type=int, default=config.tread_num, help='thread num')
    parser.add_argument('--lower', action='store_true', help='choose if lower all the words')
    args = parser.parse_args()
    print('setting:')
    print(args)
    print()

    # choose GPU and init seed
    if args.gpu >= 0:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(args.thread)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        print('using GPU device : %d' % args.gpu)
        print('GPU seed = %d' % torch.cuda.initial_seed())
        print('CPU seed = %d' % torch.initial_seed())
    else:
        use_cuda = False
        torch.set_num_threads(args.thread)
        torch.manual_seed(args.seed)
        print('CPU seed = %d' % torch.initial_seed())

    # read training , dev and test file
    print('loading three datasets...')
    train = Corpus(config.train_file[args.task], ignore_docstart=False)
    dev = Corpus(config.dev_file[args.task], ignore_docstart=False)
    test = Corpus(config.test_file[args.task], ignore_docstart=False)
    
    # collect all words, characters and labels in trainning data
    # remove words whose frequency <= 1
    vocab = Vocab(train, min_freq=1)

    # choose if use pretrained word embedding
    if args.pre_emb and config.embedding_file !=None:
        print('loading pretrained embedding...')
        pre_embedding = vocab.read_embedding(config.embedding_file)
    print('Words : %d，Characters : %d，labels : %d' %
          (vocab.num_words, vocab.num_chars, vocab.num_labels))
    save_pkl(vocab, config.vocab_file)

    # load parser featuer 
    print('loading parser feature...')
    train_parser = load_pkl(config.train_parser[args.task])
    dev_parser = load_pkl(config.dev_parser[args.task])
    test_parser = load_pkl(config.test_parser[args.task])
    print(len(train_parser),len(dev_parser),len(test_parser))
    # process training data , change string to index
    print('processing datasets...')
    train_data = process_data(vocab, train, train_parser, config.parser_layers, lower=args.lower, max_word_len=20)
    dev_data = process_data(vocab, dev, dev_parser, config.parser_layers, lower=args.lower, max_word_len=20)
    test_data = process_data(vocab, test, test_parser, config.parser_layers, lower=args.lower, max_word_len=20)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )
    dev_loader = Data.DataLoader(
        dataset=dev_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )

    # create neural network
    net = Parser_Char_LSTM_CRF(config.parser_layers, 
                               config.parser_dim, 
                               vocab.num_chars, 
                               config.char_dim, 
                               config.char_hidden, 
                               vocab.num_words,
                               config.word_dim, 
                               config.layers, 
                               config.word_hidden, 
                               vocab.num_labels, 
                               config.dropout
                               )
    if args.pre_emb:
        net.load_pretrained_embedding(pre_embedding)
    print(net)

    # if use GPU , move all needed tensors to CUDA
    if use_cuda:
        net.cuda()
        
    # init evaluator
    evaluator = Evaluator(vocab, task=args.task)
    # init trainer
    trainer = Trainer(net, config)
    # start to train
    trainer.train((train_loader, dev_loader, test_loader), evaluator)