class Corpus(object):
    def __init__(self, filename=None, ignore_docstart=False):
        self.filename = filename
        self.word_seqs, self.label_seqs = self.read(filename, ignore_docstart)
        assert(len(self.word_seqs) == len(self.label_seqs))

    @property
    def num_sentences(self):
        return len(self.word_seqs)

    @property
    def num_words(self):
        return sum(map(len, self.word_seqs))
    
    def __repr__(self):
        return '%s : sentences:%d，words:%d' % (self.filename, self.num_sentences, self.num_words)

    def __getitem(self, index):
        return zip(self.word_seqs[index], self.label_seqs[index])

    @staticmethod
    def read(filename, ignore_docstart=False):
        word_seqs = []
        label_seqs = []
        word_seq = []
        label_seq = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    # remove '-DOCSTART='
                    if ignore_docstart:
                        if word_seq[0] != '-DOCSTART-' and word_seq[0] != '-DOCSTART-'.lower():
                            word_seqs.append(word_seq)
                            label_seqs.append(label_seq)
                    else:
                        word_seqs.append(word_seq)
                        label_seqs.append(label_seq)
                    word_seq = []
                    label_seq = []
                else:
                    split = line.split()
                    word_seq.append(split[0])
                    label_seq.append(split[1])
        return word_seqs, label_seqs