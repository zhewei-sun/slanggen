# Copyright (C) 2021 Zhewei Sun

import os
import re

from collections import namedtuple

import numpy as np

from nltk.corpus import stopwords as sw
from gensim.utils import simple_preprocess

# Helper functions

punctuations = '!"#$%&()\*\+,-\./:;<=>?@[\\]^_`{|}~'

re_punc = re.compile(r"["+punctuations+r"]+")
re_space = re.compile(r" +")

stopwords = set(sw.words('english'))

Definition = namedtuple('Definition', ['word', 'type', 'def_sent', 'meta_data'])
# For slang data entries
SlangEntry = namedtuple('SlangEntry', ['word', 'def_sent', 'meta_data'])
DataIndex = namedtuple('DataIndex', ['train', 'dev', 'test'])
Triplet = namedtuple('Triplet', ['anchor', 'positive', 'negative'])

def tokenize(sentence):
    return re.compile(r"(?:^|(?<=\s))\S+(?=\s|$)").findall(sentence)

def processTokens(fun, sentence):
    return re.compile(r"(?:^|(?<=\s))\S+(?=\s|$)").sub(fun, sentence)

def normalize(array, axis=1):
    denoms = np.sum(array, axis=axis)
    if axis == 1:
        return array / denoms[:,np.newaxis]
    if axis == 0:
        return array / denoms[np.newaxis, :]
    
def normalize_L2(array, axis=1):
    if axis == 1:
        return array / np.linalg.norm(array, axis=1)[:, np.newaxis]
    if axis == 0:
        return array / np.linalg.norm(array, axis=0)[np.newaxis, :]
    
def acronym_check(entry):
    if 'acronym' in entry.def_sent:
        return True
    for c in str(entry.word):
        if ord(c) >= 65 and ord(c) <= 90:
            continue
        return False
    return True

def is_close_def(query_sent, target_sent, threshold=0.5):
    query_s = [w for w in simple_preprocess(query_sent) if w not in stopwords]
    target_s = set([w for w in simple_preprocess(target_sent) if w not in stopwords])
    overlap_c = 0
    for word in query_s:
        if word in target_s:
            overlap_c += 1
    return overlap_c >= len(query_s) * threshold

def has_close_conv_def(word, slang_def_sent, conv_data, threshold=0.5):
    conv_sents = [d['def'] for d in conv_data[word].definitions]
    for conv_sent in conv_sents:
        if is_close_def(slang_def_sent, conv_sent, threshold):
            return True
    return False

def create_directory(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)


# For conventional data entries
class Word:
    
    def __init__(self, word):
        self.word = word
        self.pos_tags = set()
        self.definitions = []

    def attach_def(self, word_def, pos, sentences):
        new_def = {'def':word_def, 'pos':pos, 'sents':sentences}
        self.pos_tags.add(pos)
        self.definitions.append(new_def)
        
# Evaluation helpers

def get_rankings(l_model, inds, labels):
    N = l_model.shape[0]
    ranks = np.zeros(l_model.shape, dtype=np.int32)
    rankings = np.zeros(N, dtype=np.int32)
        
    for i in range(N):
        ranks[i] = np.argsort(l_model[i])[::-1]
        rankings[i] = ranks[i].tolist().index(labels[inds[i]])+1
            
    return rankings
    
def get_roc(rankings, N_cat):
    roc = np.zeros(N_cat+1)
    for rank in rankings:
        roc[rank]+=1
    for i in range(1,N_cat+1):
        roc[i] = roc[i] + roc[i-1]
    return roc / rankings.shape[0]