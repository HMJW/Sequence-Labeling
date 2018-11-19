class Config(object):
    train_file = {'chunking':'../data/en/chunking/train.txt','ner':'../data/en/NER/train.txt','pos':'../data/en/pos/train.txt'}
    dev_file = {'chunking':'../data/en/chunking/dev.txt','ner':'../data/en/NER/dev.txt','pos':'../data/en/pos/dev.txt'}
    test_file = {'chunking':'../data/en/chunking/test.txt','ner':'../data/en/NER/test.txt','pos':'../data/en/pos/test.txt'}
    embedding_file = '../data/embedding/glove.6B.100d.txt'


class Char_LSTM_CRF_Config(Config):
    model = 'Char_LSTM_CRF'
    net_file = './save/char_lstm_crf.pt'
    vocab_file = './save/vocab.pkl'

    word_hidden = 300
    char_hidden = 300
    layers = 1
    dropout = 0.5
    char_dim = 30
    word_dim = 100

    optimizer = 'adam'
    epoch = 100
    lr = 0.001
    batch_size = 50
    eval_batch = 100
    decay = 0.05
    patience = 10
    shuffle = True


class Extra_LSTM_CRF_Config(Config):
    model = 'Extra_LSTM_CRF'
    train_extra = {'chunking': {'elmo':'../data/en/chunking/elmo/train-elmo.hdf5', 'parser':'../data/en/chunking/parser/train.parser', 'bert':'../data/en/chunking/bert/train.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/train-elmo.hdf5', 'parser':'../data/en/NER/parser/train.parser', 'bert':'../data/en/NER/bert/train.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/train-elmo.hdf5', 'parser':'../data/en/NER/pos/train.parser', 'bert':'../data/en/pos/bert/train.bert'}}
    dev_extra = {'chunking': {'elmo':'../data/en/chunking/elmo/dev-elmo.hdf5', 'parser':'../data/en/chunking/parser/dev.parser', 'bert':'../data/en/chunking/bert/dev.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/dev-elmo.hdf5', 'parser':'../data/en/NER/parser/dev.parser', 'bert':'../data/en/NER/bert/dev.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/dev-elmo.hdf5', 'parser':'../data/en/NER/pos/dev.parser', 'bert':'../data/en/pos/bert/dev.bert'}}
    test_extra = {'chunking': {'elmo':'../data/en/chunking/elmo/test-elmo.hdf5', 'parser':'../data/en/chunking/parser/test.parser', 'bert':'../data/en/chunking/bert/test.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/test-elmo.hdf5', 'parser':'../data/en/NER/parser/test.parser', 'bert':'../data/en/NER/bert/test.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/test-elmo.hdf5', 'parser':'../data/en/NER/pos/test.parser', 'bert':'../data/en/pos/bert/test.bert'}}
    net_file = './save/extra_lstm_crf.pt'
    vocab_file = './save/vocab.pkl'

    word_hidden = 300
    layers = 1
    dropout = 0.5
    extra_dim = {'elmo':1024,'parser':800, 'bert':768}    # elmo:1024 parser:800 bert=768
    extra_layers = {'elmo':3, 'parser':3, 'bert':4}        # elmo:3    parser:3   bert=4
    word_dim = 100

    optimizer = 'adam'
    epoch = 100
    lr = 0.001
    batch_size = 50
    eval_batch = 100
    decay = 0.05
    patience = 10
    shuffle = True


class Extra_Char_LSTM_CRF_Config(Config):
    model = 'Extra_Char_LSTM_CRF'

    train_extra = {'chunking': {'elmo':'../data/en/chunking/elmo/train-elmo.hdf5', 'parser':'../data/en/chunking/parser/train.parser', 'bert':'../data/en/chunking/bert/train.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/train-elmo.hdf5', 'parser':'../data/en/NER/parser/train.parser', 'bert':'../data/en/NER/bert/train.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/train-elmo.hdf5', 'parser':'../data/en/pos/parser/train.parser', 'bert':'../data/en/pos/bert/train.bert'}}
    dev_extra = {'chunking': {'elmo':'../data/en/chunking/elmo/dev-elmo.hdf5', 'parser':'../data/en/chunking/parser/dev.parser', 'bert':'../data/en/chunking/bert/dev.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/dev-elmo.hdf5', 'parser':'../data/en/NER/parser/dev.parser', 'bert':'../data/en/NER/bert/dev.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/dev-elmo.hdf5', 'parser':'../data/en/pos/parser/dev.parser', 'bert':'../data/en/pos/bert/dev.bert'}}
    test_extra = {'chunking': {'elmo':'../data/en/chunking/elmo/test-elmo.hdf5', 'parser':'../data/en/chunking/parser/test.parser', 'bert':'../data/en/chunking/bert/test.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/test-elmo.hdf5', 'parser':'../data/en/NER/parser/test.parser', 'bert':'../data/en/NER/bert/test.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/test-elmo.hdf5', 'parser':'../data/en/pos/parser/test.parser', 'bert':'../data/en/pos/bert/test.bert'}}
    net_file = './save/extra_char_lstm_crf.pt'
    vocab_file = './save/vocab.pkl'

    word_hidden = 300
    char_hidden = 300
    layers = 1
    dropout = 0.5
    extra_dim = {'elmo':1024, 'parser':800, 'bert':768}    # elmo:1024 parser:800 bert=768
    extra_layers = {'elmo':3, 'parser':3, 'bert':4}        # elmo:3    parser:3   bert=4
    char_dim = 30
    word_dim = 100

    optimizer = 'adam'
    epoch = 100
    lr = 0.001
    batch_size = 50
    eval_batch = 100
    decay = 0.05
    patience = 10
    shuffle = True


class Mix_Extra_LSTM_CRF_Config(Config):
    model = 'Mix_Extra_LSTM_CRF'
    train_extra1 = {'chunking': {'elmo':'../data/en/chunking/elmo/train-elmo.hdf5', 'parser':'../data/en/chunking/parser/train.parser', 'bert':'../data/en/chunking/bert/train.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/train-elmo.hdf5', 'parser':'../data/en/NER/parser/train.parser', 'bert':'../data/en/NER/bert/train.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/train-elmo.hdf5', 'parser':'../data/en/NER/pos/train.parser', 'bert':'../data/en/pos/bert/train.bert'}}
    dev_extra1 = {'chunking': {'elmo':'../data/en/chunking/elmo/dev-elmo.hdf5', 'parser':'../data/en/chunking/parser/dev.parser', 'bert':'../data/en/chunking/bert/dev.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/dev-elmo.hdf5', 'parser':'../data/en/NER/parser/dev.parser', 'bert':'../data/en/NER/bert/dev.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/dev-elmo.hdf5', 'parser':'../data/en/NER/pos/dev.parser', 'bert':'../data/en/pos/bert/dev.bert'}}
    test_extra1 = {'chunking': {'elmo':'../data/en/chunking/elmo/test-elmo.hdf5', 'parser':'../data/en/chunking/parser/test.parser', 'bert':'../data/en/chunking/bert/test.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/test-elmo.hdf5', 'parser':'../data/en/NER/parser/test.parser', 'bert':'../data/en/NER/bert/test.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/test-elmo.hdf5', 'parser':'../data/en/NER/pos/test.parser', 'bert':'../data/en/pos/bert/test.bert'}}
    train_extra2 = {'chunking': {'elmo':'../data/en/chunking/elmo/train-elmo.hdf5', 'parser':'../data/en/chunking/parser/train.parser', 'bert':'../data/en/chunking/bert/train.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/train-elmo.hdf5', 'parser':'../data/en/NER/parser/train.parser', 'bert':'../data/en/NER/bert/train.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/train-elmo.hdf5', 'parser':'../data/en/NER/pos/train.parser', 'bert':'../data/en/pos/bert/train.bert'}}
    dev_extra2 = {'chunking': {'elmo':'../data/en/chunking/elmo/dev-elmo.hdf5', 'parser':'../data/en/chunking/parser/dev.parser', 'bert':'../data/en/chunking/bert/dev.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/dev-elmo.hdf5', 'parser':'../data/en/NER/parser/dev.parser', 'bert':'../data/en/NER/bert/dev.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/dev-elmo.hdf5', 'parser':'../data/en/NER/pos/dev.parser', 'bert':'../data/en/pos/bert/dev.bert'}}
    test_extra2 = {'chunking': {'elmo':'../data/en/chunking/elmo/test-elmo.hdf5', 'parser':'../data/en/chunking/parser/test.parser', 'bert':'../data/en/chunking/bert/test.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/test-elmo.hdf5', 'parser':'../data/en/NER/parser/test.parser', 'bert':'../data/en/NER/bert/test.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/test-elmo.hdf5', 'parser':'../data/en/NER/pos/test.parser', 'bert':'../data/en/pos/bert/test.bert'}}
    net_file = './save/mix_extra_lstm_crf.pt'
    vocab_file = './save/vocab.pkl'

    word_hidden = 300
    layers = 1
    dropout = 0.5
    extra1_dim = {'elmo':1024, 'parser':800, 'bert':768}    # elmo:1024 parser:800 bert=768
    extra1_layers = {'elmo':3, 'parser':3, 'bert':4}        # elmo:3    parser:3   bert=4
    extra2_dim = {'elmo':1024, 'parser':800, 'bert':768}    # elmo:1024 parser:800 bert=768
    extra2_layers = {'elmo':3, 'parser':3, 'bert':4}        # elmo:3    parser:3   bert=4
    word_dim = 100

    optimizer = 'adam'
    epoch = 100
    lr = 0.001
    batch_size = 50
    eval_batch = 100
    decay = 0.05
    patience = 10
    shuffle = True


class Mix_Extra_Char_LSTM_CRF_Config(Config):
    model = 'Mix_Extra_Char_LSTM_CRF'
    train_extra1 = {'chunking': {'elmo':'../data/en/chunking/elmo/train-elmo.hdf5', 'parser':'../data/en/chunking/parser/train.parser', 'bert':'../data/en/chunking/bert/train.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/train-elmo.hdf5', 'parser':'../data/en/NER/parser/train.parser', 'bert':'../data/en/NER/bert/train.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/train-elmo.hdf5', 'parser':'../data/en/NER/pos/train.parser', 'bert':'../data/en/pos/bert/train.bert'}}
    dev_extra1 = {'chunking': {'elmo':'../data/en/chunking/elmo/dev-elmo.hdf5', 'parser':'../data/en/chunking/parser/dev.parser', 'bert':'../data/en/chunking/bert/dev.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/dev-elmo.hdf5', 'parser':'../data/en/NER/parser/dev.parser', 'bert':'../data/en/NER/bert/dev.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/dev-elmo.hdf5', 'parser':'../data/en/NER/pos/dev.parser', 'bert':'../data/en/pos/bert/dev.bert'}}
    test_extra1 = {'chunking': {'elmo':'../data/en/chunking/elmo/test-elmo.hdf5', 'parser':'../data/en/chunking/parser/test.parser', 'bert':'../data/en/chunking/bert/test.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/test-elmo.hdf5', 'parser':'../data/en/NER/parser/test.parser', 'bert':'../data/en/NER/bert/test.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/test-elmo.hdf5', 'parser':'../data/en/NER/pos/test.parser', 'bert':'../data/en/pos/bert/test.bert'}}
    train_extra2 = {'chunking': {'elmo':'../data/en/chunking/elmo/train-elmo.hdf5', 'parser':'../data/en/chunking/parser/train.parser', 'bert':'../data/en/chunking/bert/train.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/train-elmo.hdf5', 'parser':'../data/en/NER/parser/train.parser', 'bert':'../data/en/NER/bert/train.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/train-elmo.hdf5', 'parser':'../data/en/NER/pos/train.parser', 'bert':'../data/en/pos/bert/train.uncased-aver'}}
    dev_extra2 = {'chunking': {'elmo':'../data/en/chunking/elmo/dev-elmo.hdf5', 'parser':'../data/en/chunking/parser/dev.parser', 'bert':'../data/en/chunking/bert/dev.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/dev-elmo.hdf5', 'parser':'../data/en/NER/parser/dev.parser', 'bert':'../data/en/NER/bert/dev.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/dev-elmo.hdf5', 'parser':'../data/en/NER/pos/dev.parser', 'bert':'../data/en/pos/bert/dev.bert'}}
    test_extra2 = {'chunking': {'elmo':'../data/en/chunking/elmo/test-elmo.hdf5', 'parser':'../data/en/chunking/parser/test.parser', 'bert':'../data/en/chunking/bert/test.uncased-aver'},
                    'ner': {'elmo':'../data/en/NER/elmo/test-elmo.hdf5', 'parser':'../data/en/NER/parser/test.parser', 'bert':'../data/en/NER/bert/test.uncased'},
                    'pos': {'elmo':'../data/en/pos/elmo/test-elmo.hdf5', 'parser':'../data/en/NER/pos/test.parser', 'bert':'../data/en/pos/bert/test.bert'}}
    net_file = './save/mix_extra_char_lstm_crf.pt'
    vocab_file = './save/vocab.pkl'

    word_hidden = 300
    char_hidden = 300
    layers = 1
    dropout = 0.5
    extra1_dim = {'elmo':1024, 'parser':800, 'bert':768}    # elmo:1024 parser:800 bert=768
    extra1_layers = {'elmo':3, 'parser':3, 'bert':4}        # elmo:3    parser:3   bert=4
    extra2_dim = {'elmo':1024, 'parser':800, 'bert':768}    # elmo:1024 parser:800 bert=768
    extra2_layers = {'elmo':3, 'parser':3, 'bert':4}        # elmo:3    parser:3   bert=4
    word_dim = 100
    char_dim = 30

    optimizer = 'adam'
    epoch = 100
    lr = 0.001
    batch_size = 50
    eval_batch = 100
    decay = 0.05
    patience = 10
    shuffle = True


config = {
    'char_lstm_crf' : Char_LSTM_CRF_Config,
    'extra_lstm_crf' : Extra_LSTM_CRF_Config,
    'extra_char_lstm_crf' : Extra_Char_LSTM_CRF_Config,
    'mix_extra_lstm_crf' : Mix_Extra_LSTM_CRF_Config,
    'mix_extra_char_lstm_crf' : Mix_Extra_Char_LSTM_CRF_Config
}