#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import json
import re
from nltk.tokenize import word_tokenize
p = '!\"#$%&\'()*+,-.:;=?@[\]^_`{|}~'


def encode_entity(sentence, indices):
    sentence = encode(sentence.split(), indices)
    sentence = sentence.translate(str.maketrans('', '', p))
        
    assert '<e1>' in sentence
    assert '<e2>' in sentence
    assert '</e1>' in sentence
    assert '</e2>' in sentence
    
    return sentence.rstrip().lower().split()

def encode(toks, indices):
    for i in range(len(indices)):
      toks[indices[i]] = f'<e{i+1}> {toks[indices[i]]} </e{i+1}>'
    return " ".join(toks)


def convert(path_src, path_des):
    with open(path_src, 'r', encoding='utf-8') as fr:
        with open(path_des, 'w+', encoding='utf-8') as fw:
            for ids , line in enumerate(fr.readlines()):
                label, ind1, ind2, sentence = line.split('\t')
                indices = [int(ind1), int(ind2)]
                sentence = encode_entity(sentence.rstrip(), indices)
                meta = dict(
                    id=ids+1,
                    relation=label,
                    sentence=sentence,
                    indices=indices
                    )
                json.dump(meta, fw, ensure_ascii=False)
                fw.write('\n')


if __name__ == '__main__':
    path_train = './SEMEVAL_TRAIN.TSV'
    path_test = './SEMEVAL_TEST.TSV'

    convert(path_train, 'train.json')
    convert(path_test, 'test.json')
