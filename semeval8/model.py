#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn, rnn, contrib
from mxnet import autograd as ag
from mxnet.gluon.loss import L2Loss

from mxnet.gluon.block import HybridBlock, Block
from gluonnlp.model import AttentionCell, MLPAttentionCell, DotProductAttentionCell, MultiHeadAttentionCell

class RelationClassifier(HybridBlock):
    def __init__(self, word_vec, class_num, config):
        super(RelationClassifier, self).__init__(prefix=prefix, params=params)
        
        self.word_vec = word_vec
        self.class_num = class_num

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.hidden_size = config.hidden_size
        self.layers_num = config.layers_num
        self.emb_dropout_value = config.emb_dropout
        self.lstm_dropout_value = config.lstm_dropout
        self.linear_dropout_value = config.linear_dropout

        # net structures and operations
        with self.name_scope():
            self.word_embedding = nn.Embedding(self.word_vec.shape)
        
            self.lstm = rnn.LSTM(
                            input_size=self.word_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.layers_num,
                            bias=True,
                            layout='NTC',
                            dropout=0,
                            bidirectional=True,
                            )
                            
            self.lstm_layer = self._get_lstm_layer()
            self.attention_layer = self._get_attention_layer()
            
            self.tanh = nn.Activation('tanh')
            self.emb_dropout = nn.Dropout(self.emb_dropout_value)
            self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)
            self.linear_dropout = nn.Dropout(self.linear_dropout_value)

            self.att_weight = nn.Parameter(torch.randn(1, self.hidden_size, 1))
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(nn.Dense(units=256, flatten=True, activation='softrelu'))
                self.output.add(nn.Dense(units=256, activation='softrelu'))
                self.output.add(nn.Dense(
                                    self.class_num,
                                    in_units=self.hidden_size,
                                    use_bias=True
                                    ))
                
        # initialize weight
        self.initialize(mx.init.Xavier(magnitude=2.34), ctx=config.device, force_reinit=False)


    def _get_lstm_layer(self,**kwargs):
        
        # # mask : (seq_len, batch_size, 1)
        mask[mask==0] = float('-inf')
        mask = nd.expand_dims(mask, axis=2)        
        h, _ = self.lstm(x) * mask
        print(h.shape)
        h = h.reshape((-1, self.max_len, 2, self.hidden_size))
        h = mx.nd.sum(h, axis=2)  # B*L*H
        return h

    def _get_attention_layer(self,**kwargs):
        att_weight = self.att_weight.reshape(mask.shape[0], self.hidden_size, -1)  # B*H*1
        att_score = mx.nd.linalg_gemm2(self.tanh(h), att_weight)  # B*L*H  *  B*H*1 -> B*L*1

        # mask, remove the effect of 'PAD'
        mask[mask==0] = float('-inf')
        mask = nd.expand_dims(mask, axis=2)          # B*L*1
        att_score = att_score * mask  # B*L*1
        att_weight = F.softmax(att_score, dim=1)  # B*L*1

        reps = mx.nd.linalg_gemm2(h.transpose(1, 2), att_weight).squeeze(dim=-1)  # B*H*L *  B*L*1 -> B*H*1 -> B*H
        reps = self.tanh(reps)  # B*H
        return reps

    def hybrid_forward(self, data):
        token = data[:, 0, :].reshape((-1, self.max_len))
        mask = data[:, 1, :].reshape((-1, self.max_len))
        lengths = data[:, 2, :]
        
        emb = self.word_embedding(token)  # B*L*word_dim
        emb = self.emb_dropout(emb)
        h = self.lstm_layer(emb, mask)  # B*L*H
        h = self.lstm_dropout(h)
        reps = self.attention_layer(h, mask)  # B*reps
        reps = self.linear_dropout(reps)
        logits = self.output(reps)
        return logits
