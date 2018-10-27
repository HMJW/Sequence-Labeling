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
    gpu = -1
    lr = 0.001
    batch_size = 50
    eval_batch = 100
    tread_num = 4
    decay = 0.05
    patience = 10
    shuffle = True


class Elmo_LSTM_CRF_Config(Config):
    model = 'Elmo_LSTM_CRF'
    train_elmo = {'chunking':'../data/en/chunking/train-elmo.hdf5','ner':'../data/en/NER/train-elmo.hdf5','pos':'../data/en/pos/train-elmo.hdf5'}
    dev_elmo = {'chunking':'../data/en/chunking/dev-elmo.hdf5','ner':'../data/en/NER/dev-elmo.hdf5','pos':'../data/en/pos/dev-elmo.hdf5'}
    test_elmo = {'chunking':'../data/en/chunking/test-elmo.hdf5','ner':'../data/en/NER/test-elmo.hdf5','pos':'../data/en/pos/test-elmo.hdf5'}
    net_file = './save/elmo_lstm_crf.pt'
    vocab_file = './save/vocab.pkl'

    word_hidden = 300
    char_hidden = 300
    layers = 1
    dropout = 0.5
    elmo_dim = 1024
    elmo_layers = 3
    char_dim = 30
    word_dim = 100

    optimizer = 'adam'
    epoch = 100
    gpu = -1
    lr = 0.001
    batch_size = 50
    eval_batch = 100
    tread_num = 4
    decay = 0.05
    patience = 10
    shuffle = True


class Parser_Char_LSTM_CRF_Config(Config):
    model = 'Parser_Char_LSTM_CRF'
    train_parser = {'chunking':'../data/en/chunking/train.parser','ner':'../data/en/NER/train.parser','pos':'../data/en/pos/train.parser'}
    dev_parser = {'chunking':'../data/en/chunking/dev.parser','ner':'../data/en/NER/dev.parser','pos':'../data/en/pos/dev.parser'}
    test_parser = {'chunking':'../data/en/chunking/test.parser','ner':'../data/en/NER/test.parser','pos':'../data/en/pos/test.parser'}
    net_file = './save/parser_char_lstm_crf.pt'
    vocab_file = './save/vocab.pkl'

    word_hidden = 300
    char_hidden = 300
    layers = 1
    dropout = 0.5
    parser_dim = 800
    parser_layers = 3
    char_dim = 30
    word_dim = 100

    optimizer = 'adam'
    epoch = 100
    gpu = -1
    lr = 0.001
    batch_size = 50
    eval_batch = 100
    tread_num = 4
    decay = 0.05
    patience = 10
    shuffle = True


class Parser_Elmo_LSTM_CRF_Config(Config):
    model = 'Parser_Elmo_LSTM_CRF'
    train_elmo = {'chunking':'../data/en/chunking/train-elmo.hdf5','ner':'../data/en/NER/train-elmo.hdf5','pos':'../data/en/pos/train-elmo.hdf5'}
    dev_elmo = {'chunking':'../data/en/chunking/dev-elmo.hdf5','ner':'../data/en/NER/dev-elmo.hdf5','pos':'../data/en/pos/dev-elmo.hdf5'}
    test_elmo = {'chunking':'../data/en/chunking/test-elmo.hdf5','ner':'../data/en/NER/test-elmo.hdf5','pos':'../data/en/pos/test-elmo.hdf5'}
    train_parser = {'chunking':'../data/en/chunking/train.parser','ner':'../data/en/NER/train.parser','pos':'../data/en/pos/train.parser'}
    dev_parser = {'chunking':'../data/en/chunking/dev.parser','ner':'../data/en/NER/dev.parser','pos':'../data/en/pos/dev.parser'}
    test_parser = {'chunking':'../data/en/chunking/test.parser','ner':'../data/en/NER/test.parser','pos':'../data/en/pos/test.parser'}
    net_file = './save/parser_char_lstm_crf.pt'
    vocab_file = './save/vocab.pkl'

    word_hidden = 300
    char_hidden = 300
    layers = 1
    dropout = 0.5
    elmo_dim = 1024
    elmo_layers = 3
    parser_dim = 800
    parser_layers = 3
    char_dim = 30
    word_dim = 100

    optimizer = 'adam'
    epoch = 100
    gpu = -1
    lr = 0.001
    batch_size = 50
    eval_batch = 100
    tread_num = 4
    decay = 0.05
    patience = 10
    shuffle = True


config = {
    'char_lstm_crf' : Char_LSTM_CRF_Config,
    'elmo_lstm_crf' : Elmo_LSTM_CRF_Config,
    'parser_char_lstm_crf' : Parser_Char_LSTM_CRF_Config,
    'parser_elmo_lstm_crf' : Parser_Elmo_LSTM_CRF_Config
}