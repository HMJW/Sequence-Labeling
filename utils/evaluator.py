import torch


class Decoder(object):
    @staticmethod
    def viterbi(crf, emit_matrix):
        '''
        viterbi for one sentence
        '''
        length = emit_matrix.size(0)
        max_score = torch.zeros_like(emit_matrix)
        paths = torch.zeros_like(emit_matrix, dtype=torch.long)

        max_score[0] = emit_matrix[0] + crf.strans
        for i in range(1, length):
            emit_scores = emit_matrix[i]
            scores = emit_scores + crf.transitions + \
                max_score[i - 1].view(-1, 1).expand(-1, crf.labels_num)
            max_score[i], paths[i] = torch.max(scores, 0)

        max_score[-1] += crf.etrans
        prev = torch.argmax(max_score[-1])
        predict = [prev.item()]
        for i in range(length - 1, 0, -1):
            prev = paths[i][prev.item()]
            predict.insert(0, prev.item())
        return torch.tensor(predict)
    
    @staticmethod
    def viterbi_batch(crf, emits, masks):
        '''
        viterbi for sentences in batch
        '''
        sen_len, batch_size, labels_num = emits.shape

        lens = masks.sum(dim=0)  # [batch_size]
        scores = torch.zeros_like(emits)  # [sen_len, batch_size, labels_num]
        paths = torch.zeros_like(emits, dtype=torch.long) # [sen_len, batch_size, labels_num]

        scores[0] = crf.strans + emits[0]  # [batch_size, labels_num]
        for i in range(1, sen_len):
            trans_i = crf.transitions.unsqueeze(0)  # [1, labels_num, labels_num]
            emit_i = emits[i].unsqueeze(1)  # [batch_size, 1, labels_num]
            score = scores[i - 1].unsqueeze(2)  # [batch_size, labels_num, 1]
            score_i = trans_i + emit_i + score  # [batch_size, labels_num, labels_num]
            scores[i], paths[i] = torch.max(score_i, dim=1)

        predicts = []
        for i, length in enumerate(lens):
            prev = torch.argmax(scores[length - 1, i] + crf.etrans)
            predict = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                predict.append(prev)
            predicts.append(torch.tensor(predict).flip(0))

        return predicts


class Evaluator(object):
    def __init__(self, vocab, task='chunking'):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0
        self.vocab = vocab
        self.task = task

    def clear_num(self):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0

    def cal_num(self, pred, gold):
        set1 = self.recognize(pred)
        set2 = self.recognize(gold)
        intersction = set1 & set2
        correct_num = len(intersction)
        pred_num = len(set1)
        gold_num = len(set2)
        return correct_num, pred_num, gold_num

    def recognize(self, sequence):
        """
        copy from the paper
        """
        chunks = []
        current = None

        for i, label in enumerate(sequence):
            if label.startswith('B-'):

                if current is not None:
                    chunks.append('@'.join(current))
                current = [label.replace('B-', ''), '%d' % i]

            elif label.startswith('S-'):

                if current is not None:
                    chunks.append('@'.join(current))
                    current = None
                base = label.replace('S-', '')
                chunks.append('@'.join([base, '%d' % i]))

            elif label.startswith('I-'):

                if current is not None:
                    base = label.replace('I-', '')
                    if base == current[0]:
                        current.append('%d' % i)
                    else:
                        chunks.append('@'.join(current))
                        current = [base, '%d' % i]

                else:
                    current = [label.replace('I-', ''), '%d' % i]

            elif label.startswith('E-'):

                if current is not None:
                    base = label.replace('E-', '')
                    if base == current[0]:
                        current.append('%d' % i)
                        chunks.append('@'.join(current))
                        current = None
                    else:
                        chunks.append('@'.join(current))
                        current = [base, '%d' % i]
                        chunks.append('@'.join(current))
                        current = None

                else:
                    current = [label.replace('E-', ''), '%d' % i]
                    chunks.append('@'.join(current))
                    current = None
            else:
                if current is not None:
                    chunks.append('@'.join(current))
                current = None

        if current is not None:
            chunks.append('@'.join(current))

        return set(chunks)

    def eval(self, network, data_loader):
        network.eval()
        total_loss = 0.0

        for batch in data_loader:
            batch_size = batch[0].size(0)
            # mask = torch.arange(y.size(1)) < lens.unsqueeze(-1)
            # mask = word_idxs.gt(0)
            mask, out, targets = network.forward_batch(batch)
            sen_lens = mask.sum(1)

            batch_loss = network.get_loss(out.transpose(0, 1), targets.t(), mask.t())
            total_loss += batch_loss * batch_size

            predicts = Decoder.viterbi_batch(network.crf, out.transpose(0, 1), mask.t())
            targets = torch.split(targets[mask], sen_lens.tolist())
            
            if self.task == 'pos':
                for predict, target in zip(predicts, targets):
                    predict = predict.tolist()
                    target = target.tolist()                
                    correct_num = sum(x==y for x,y in zip(predict, target))
                    self.correct_num += correct_num
                    self.pred_num += len(predict)
                    self.gold_num += len(target)
            elif self.task == 'chunking' or self.task == 'ner':
                for predict, target in zip(predicts, targets):
                    predict = self.vocab.id2label(predict.tolist())
                    target = self.vocab.id2label(target.tolist())
                    correct_num, pred_num, gold_num = self.cal_num(predict, target)
                    self.correct_num += correct_num
                    self.pred_num += pred_num
                    self.gold_num += gold_num

        if self.task == 'pos':
            precision = self.correct_num/self.pred_num
            self.clear_num()
            return total_loss, precision
        elif self.task == 'chunking' or self.task == 'ner':
            precision = self.correct_num/self.pred_num
            recall = self.correct_num/self.gold_num
            Fscore = (2*precision*recall)/(precision+recall)
            self.clear_num()
            return total_loss, precision, recall, Fscore
