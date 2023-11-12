#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import argparse
import mxnet as mx
import os
import random
import json
import numpy as np
from mxnet import nd

class Config(object):
    def __init__(self):
        # get init config
        args = self.__get_config()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

        # select device
        self.device = None
        if self.cuda >= 0 and mx.context.num_gpus():
            try:
                ctx = mx.gpu(self.cuda)
                _ = nd.array([0], ctx=ctx)
                print('device: \n', _)
                self.device = ctx
            except:
                self.device = mx.cpu()

        # determine the model name and model dir
        if self.model_name is None:
            self.model_name = 'BiLSTM_ATT_Model'
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # backup data
        self.__config_backup(args)

        # set the random seed
        self.__set_seed(self.seed)

    def __get_config(self):
        parser = argparse.ArgumentParser()
        parser.description = 'config for models'

        # several key selective parameters
        parser.add_argument('--data_dir', type=str,
                            default='./data',
                            help='dir to load data')
        parser.add_argument('--output_dir', type=str,
                            default='./output',
                            help='dir to save output')

        # word embedding
        parser.add_argument('--embedding_path', type=str,
                            default='./embedding/glove.6B.100d.txt',
                            help='pre_trained word embedding')
        parser.add_argument('--word_dim', type=int,
                            default=100,
                            help='dimension of word embedding')

        # train settings
        parser.add_argument('--model_name', type=str,
                            default=None,
                            help='model name')
        parser.add_argument('--mode', type=int,
                            default=1,
                            choices=[0, 1],
                            help='running mode: 1 for training; otherwise testing')
        parser.add_argument('--seed', type=int,
                            default=5782,
                            help='random seed')
        parser.add_argument('--cuda', type=int,
                            default=1,
                            help='num of gpu device, if -1, select cpu')
        parser.add_argument('--epoch', type=int,
                            default=30,
                            help='max epoches during training')

        # hyper parameters
        parser.add_argument('--batch_size', type=int,
                            default=10,
                            help='batch size')
        parser.add_argument('--lr', type=float,
                            default=0.05,
                            help='learning rate')
        parser.add_argument('--max_len', type=int,
                            default=100,
                            help='max length of sentence')

        parser.add_argument('--emb_dropout', type=float,
                            default=0.3,
                            help='the possiblity of dropout in embedding layer')
        parser.add_argument('--lstm_dropout', type=float,
                            default=0.3,
                            help='the possiblity of dropout in (Bi)LSTM layer')
        parser.add_argument('--linear_dropout', type=float,
                            default=0.5,
                            help='the possiblity of dropout in liner layer')
        parser.add_argument('--hidden_size', type=int,
                            default=100,
                            help='the dimension of hidden units in (Bi)LSTM layer')
        parser.add_argument('--layers_num', type=int,
                            default=3,
                            help='num of RNN layers')

        args = parser.parse_args()
        return args

    def __set_seed(self, seed=1234):
        os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
        mx.random.seed(seed)

    def __config_backup(self, args):
        config_backup_path = os.path.join(self.model_dir, 'config.json')
        with open(config_backup_path, 'w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, ensure_ascii=False)

    def print_config(self):
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])


if __name__ == '__main__':
    config = Config()
    config.print_config()
