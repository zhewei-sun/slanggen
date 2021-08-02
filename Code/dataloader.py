# Copyright (C) 2021 Zhewei Sun

import abc

import numpy as np

from .util import *

class ConvDataset:
    
    def __init__(self, data_path):
        self.entries, self.vocab = self.load_data(data_path)
        
        self.N_total = 0
        for e in self.entries:
            self.N_total += len(e.definitions)
        self.V = len(self.vocab)
        
        self.data = {d.word:d for d in self.entries}
    
    @abc.abstractmethod
    def load_data(self, data_path):
        raise NotImplementedError()
        
    def __str__(self):
        out = ""
        out += "Dataset Name: " + "\n"
        out += "Total Definition Entries: %d" % self.N_total + "\n"
        out += "Vocab Size: %d" % self.V + "\n"
        return out
    
class WN_Dataset(ConvDataset):
    
    def load_data(self, data_path):
        data_WN = np.load(data_path, allow_pickle=True)
        vocab_WN = set([w.word for w in data_WN])
        return data_WN, vocab_WN

class SlangDataset:
    
    def __init__(self, slang_path, conv_dataset):
        
        self.meta_set = set()
        
        self.slang_data, self.conv_data = self.process_entries(slang_path, conv_dataset)
        
        self.N_total = len(self.slang_data)
        
        vocab = []
        vocab_set = set()
        for d in self.slang_data:
            word = str(d.word)
            if word not in vocab_set:
                vocab.append(word)
                vocab_set.add(word)
        self.vocab = np.asarray(vocab)
        self.word2id = {self.vocab[i]:i for i in range(len(self.vocab))}
        self.vocab_ids = np.asarray([self.word2id[self.slang_data[i].word] for i in range(self.N_total)], dtype=np.int64)
        
        self.V = len(self.vocab)
        
    def has_meta(self, attribute):
        return attribute in self.meta_set
        
    @abc.abstractmethod
    def load_data(self, slang_path, conv_dataset):
        raise NotImplementedError()
        
    def process_entries(self, slang_path, conv_dataset):
        slang_entries = self.load_data(slang_path, conv_dataset)
        
        conv_data = conv_dataset.data
        
        slang_data = [d for d in slang_entries if not acronym_check(d)]
        slang_data = [d for d in slang_data if not has_close_conv_def(str(d.word), d.def_sent, conv_data)]
        
        return slang_data, conv_data
        
    def __str__(self):
        out = ""
        out += "Dataset Name: " + "\n"
        out += "Total Definition Entries: %d" % self.N_total + "\n"
        out += "Vocab Size: %d" % self.V + "\n"
        return out
    
class Urban_Dataset(SlangDataset):
    
    def __init__(self, slang_path, conv_dataset):
        super().__init__(slang_path, conv_dataset)
        
    def load_data(self, slang_path, conv_dataset):
        
        data_Urban_raw = np.load(slang_path, allow_pickle=True)
        
        re_hex = re.compile(r"\\x[a-f0-9][a-f0-9]")
        re_spacechar = re.compile(r"\\(n|t)")
        
        def process_def(d):
            return SlangEntry(d[0], re_spacechar.sub('', re_hex.sub('', d[1])), {})
        
        data_Urban = [process_def(d) for d in data_Urban_raw if d[0] in conv_dataset.vocab]
        
        return data_Urban