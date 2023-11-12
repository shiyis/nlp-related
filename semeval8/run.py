#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import os
from config import Config
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd

from utils import WordEmbeddingLoader, RelationLoader, SemEvalDataLoader
from model import RelationClassifier
from evaluate import Eval

PATH = os.path.realpath(config.model_dir)
assert os.path.isdir(PATH)

def print_result(predict_label, id2rel, start_idx=8001):
    with open('predicted_result.txt', 'w', encoding='utf-8') as fw:
        for i in range(0, predict_label.shape[0]):
            fw.write('{}\t{}\n'.format(
                start_idx+i, id2rel[int(predict_label[i])]))


def train(model, criterion, loader, config):
    #loead data
    train_loader, dev_loader, _ = loader
    
    

    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    for p in model.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)
    
    optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': config.lr})

    print(model)
    print('traning model parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    print('--------------------------------------')
    print('start to train the model ...')

    eval_tool = Eval(config)
    max_f1 = -float('inf')
    for epoch in range(1, config.epoch+1):
        for step, (data, label) in enumerate(train_loader):
            model.train()
            data = data.as_in_ctx(config.device)
            label = label.as_in_ctx(config.device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=5)
            optimizer.step()

        _, train_loss, _ = eval_tool.evaluate(model, criterion, train_loader)
        f1, dev_loss, _ = eval_tool.evaluate(model, criterion, dev_loader)

        print('[%03d] train_loss: %.3f | dev_loss: %.3f | micro f1 on dev: %.4f'
              % (epoch, train_loss, dev_loss, f1), end=' ')
        if f1 > max_f1:
            max_f1 = f1
            model.save_parameters(f'{PATH}model.pkl')
            print('>>> save model!')
        else:
            print()


def test(model, criterion, loader, config):
    print('--------------------------------------')
    print('start test ...')

    _, _, test_loader = loader
    model.load_state_dict(torch.load(
        os.path.join(config.model_dir, 'model.pkl')))
    eval_tool = Eval(config)
    f1, test_loss, predict_label = eval_tool.evaluate(
        model, criterion, test_loader)
    print('test_loss: %.3f | micro f1 on test:  %.4f' % (test_loss, f1))
    return predict_label


if __name__ == '__main__':
    config = Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    loader = SemEvalDataLoader(rel2id, word2id, config)

    train_loader, dev_loader = None, None
    if config.mode == 1:  # train mode
        train_loader = loader.get_train()
        dev_loader = loader.get_dev()
    test_loader = loader.get_test()
    loader = [train_loader, dev_loader, test_loader]
    print('finish!')

    print('--------------------------------------')
    model = Att_BLSTM(word_vec=word_vec, class_num=class_num, config=config)
    model.initialize(mx.init.Xavier(magnitude=2.34), ctx=config.device)
    loss_fn = nn.CrossEntropyLoss()

    if config.mode == 1:  # train mode
        train(model, loss_fn, loader, config)
    predict_label = test(model, loss_fn, loader, config)
    print_result(predict_label, id2rel)
