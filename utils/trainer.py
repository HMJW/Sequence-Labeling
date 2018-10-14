import datetime

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer(object):
    def __init__(self, network, config):
        self.network = network
        self.config = config
        # choose optimizer
        if config.optimizer == 'sgd':
            print('Using SGD optimizer...')
            self.optimizer = optim.SGD(network.parameters(), lr=config.lr)
        elif config.optimizer == 'adam':
            print('Using Adam optimizer...')
            self.optimizer = optim.Adam(network.parameters(), lr=config.lr)

    def lr_decay(self, optimizer, epoch, decay_rate, init_lr):
        lr = init_lr/(1+decay_rate*epoch)
        print("Learning rate is set as:", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, data_loaders, evaluator):
        # record some parameters
        max_precision = 0
        test_precision = 0
        max_epoch = 0
        patience = 0

        train_loader, dev_loader, test_loader = data_loaders
        train_num = len(train_loader.dataset)
        dev_num = len(dev_loader.dataset)
        test_num = len(test_loader.dataset)

        # begin to train
        print('start to train the model ')
        for e in range(self.config.epoch):
            print('==============================Epoch<%d>==============================' % (e + 1))
            self.network.train()
            time_start = datetime.datetime.now()

            if self.config.optimizer == 'sgd':
                self.lr_decay(self.optimizer, e, self.config.decay, self.config.lr)
                
            for batch in train_loader:
                self.optimizer.zero_grad()
                # mask = torch.arange(label_idxs.size(1)) < lens.unsqueeze(-1)
                # mask = word_idxs.gt(0)
                mask, out, targets = self.network.forward_batch(batch)
                loss = self.network.get_loss(out.transpose(0,1), targets.t(), mask.t())
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                self.optimizer.step()

            with torch.no_grad():
                if evaluator.task == 'chunking' or evaluator.task == 'ner':
                    train_total_loss, train_p, train_r, train_F = evaluator.eval(self.network, train_loader)
                    print('train : loss = %.4f  precision = %.4f  recall = %.4f  F = %.4f' %
                        (train_total_loss/train_num, train_p, train_r, train_F))

                    dev_total_loss, dev_p, dev_r, dev_F = evaluator.eval(self.network, dev_loader)
                    print('dev   : loss = %.4f  precision = %.4f  recall = %.4f  F = %.4f' %
                            (dev_total_loss/dev_num, dev_p, dev_r, dev_F))

                    test_total_loss, test_p, test_r, test_F = evaluator.eval(self.network, test_loader)
                    print('test  : loss = %.4f  precision = %.4f  recall = %.4f  F = %.4f' %
                            (test_total_loss/test_num, test_p, test_r, test_F))
                elif evaluator.task == 'pos':
                    train_total_loss, train_p = evaluator.eval(self.network, train_loader)
                    print('train : loss = %.4f  precision = %.4f' % (train_total_loss/train_num, train_p))

                    dev_total_loss, dev_p = evaluator.eval(self.network, dev_loader)
                    print('dev   : loss = %.4f  precision = %.4f' % (dev_total_loss/dev_num, dev_p))

                    test_total_loss, test_p = evaluator.eval(self.network, test_loader)
                    print('test  : loss = %.4f  precision = %.4f' % (test_total_loss/test_num, test_p))
                
            # save the model when dev precision get better
            if evaluator.task == 'chunking' or evaluator.task == 'ner':
                if dev_F > max_precision:
                    max_precision = dev_F
                    test_precision = test_F
                    max_epoch = e + 1
                    patience = 0
                    print('save the model...')
                    torch.save(self.network.state_dict(), self.config.net_file)
                else:
                    patience += 1
            elif evaluator.task == 'pos':
                if dev_p > max_precision:
                    max_precision = dev_p
                    test_precision = test_p
                    max_epoch = e + 1
                    patience = 0
                    print('save the model...')
                    torch.save(self.network.state_dict(), self.config.net_file)
                else:
                    patience += 1

            time_end = datetime.datetime.now()
            print('iter executing time is ' + str(time_end - time_start) + '\n')
            if patience > self.config.patience:
                break

        print('train finished with epoch: %d / %d' % (e + 1, self.config.epoch))
        if evaluator.task == 'chunking' or evaluator.task == 'ner':
            print('best epoch is epoch = %d ,the dev F = %.4f the test F = %.4f' %
                (max_epoch, max_precision, test_precision))
        elif evaluator.task == 'pos':
            print('best epoch is epoch = %d ,the dev precision = %.4f the test precision = %.4f' %
                (max_epoch, max_precision, test_precision))
        print(str(datetime.datetime.now()))
