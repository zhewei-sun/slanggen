# Copyright (C) 2021 Zhewei Sun

import numpy as np

from CatGO.categorize import Categorizer
from tqdm import trange

import torch

from .util import *

class SlangGenModel:
    
    DEFAULT_PARAMS = {'embed_name':'SBERT_contrastive', 'out_name':'predictions', 'model':'cf_prototype_5', 'prior_name':'uniform', 'contr_params':None}
    
    def __init__(self, trainer, data_dir='', verbose=False):
        
        self.trainer = trainer
        self.data_dir = data_dir
        self.verbose=verbose
        
        self.vocab = trainer.dataset.vocab
        self.labels = trainer.dataset.vocab_ids
        
        self.cf_feats = np.stack([self.trainer.word_dist], axis=0)
        
    def train_contrastive(self, slang_ind, fold_name='default', params=None):
        
        if params is None:
            params = self.DEFAULT_PARAMS
            
        self.trainer.preprocess_slang_data(slang_ind, fold_name=fold_name)
        self.trainer.train_contrastive_model(slang_ind, fold_name=fold_name, params=params['contr_params'])
        self.trainer.get_trained_embeddings(slang_ind, fold_name=fold_name)
        
    def train_categorization(self, slang_ind, fold_name='default', params=None):
        
        if params is None:
            params = self.DEFAULT_PARAMS
        
        data_dir = self.data_dir + '/' + fold_name + '/'
        
        def_embeds = np.load(data_dir+'sum_embed_'+params['embed_name']+'.npz')
        E = def_embeds['train'].shape[1]
        
        conv_embed = def_embeds['standard']
        
        vocab_embeds = self.load_examplar_embeddings(fold_name=fold_name, def_embeds=def_embeds, params=params)
            
        train_dev_inds = np.concatenate((slang_ind.train, slang_ind.dev))
            
        slang_def_embeds = np.concatenate([def_embeds['train'], def_embeds['dev']])
        labels = self.labels[train_dev_inds]
        
        categorizer = Categorizer(self.vocab, vocab_embeds, self.cf_feats)
        
        model_dir = data_dir+params['out_name']+'/'
        create_directory(model_dir)
        categorizer.set_datadir(model_dir)
        
        if params['prior_name'] != 'uniform':
            categorizer.add_prior(params['prior_name'], params['prior'])
        
        categorizer.run_categorization(slang_def_embeds, labels, models=[params['model']], prior=params['prior_name'], mode='train', verbose=self.verbose)
    
    def load_examplar_embeddings(self, fold_name='default', def_embeds=None, params=None):
        
        if params is None:
            params = self.DEFAULT_PARAMS
            
        data_dir = self.data_dir + '/' + fold_name + '/'
            
        if def_embeds is None:
            def_embeds = np.load(data_dir+'sum_embed_'+params['embed_name']+'.npz')
        
        E = def_embeds['train'].shape[1]
        
        conv_embed = def_embeds['standard']
        
        vocab_embeds = []
        c = 0
        for i in range(self.trainer.dataset.V):
            num_def = len(self.trainer.dataset.conv_data[self.vocab[i]].definitions)
            embed = np.zeros((num_def, E))
            for j in range(num_def):        
                embed[j,:] = conv_embed[c,:]
                c += 1
            vocab_embeds.append(embed)
            
        return vocab_embeds
        
    def predict_testset(self, slang_ind, fold_name='default', params=None):
        
        if params is None:
            params = self.DEFAULT_PARAMS
        
        data_dir = self.data_dir + '/' + fold_name + '/'
        
        def_embeds = np.load(data_dir+'sum_embed_'+params['embed_name']+'.npz')
        E = def_embeds['train'].shape[1]
        
        slang_def_embeds = def_embeds['test']
        labels = self.labels[slang_ind.test]
        
        self.predict(slang_def_embeds, labels, fold_name=fold_name, mode='test', params=None)
        
    def predict_from_definitions(self, slang_def_sents, labels, fold_name='default', mode='test', params=None):
        
        if params is None:
            params = self.DEFAULT_PARAMS
        
        data_dir = self.data_dir + '/' + fold_name + '/'
        
        slang_def_embeds = self.trainer.get_testtime_embeddings(slang_def_sents, fold_name=fold_name)
        
        self.predict(slang_def_embeds, labels, fold_name=fold_name, mode=mode, params=None)
        
    def predict(self, slang_def_embeds, labels, fold_name='default', mode='test', params=None):
        
        if params is None:
            params = self.DEFAULT_PARAMS
        
        data_dir = self.data_dir + '/' + fold_name + '/'
        
        vocab_embeds = self.load_examplar_embeddings(fold_name=fold_name, params=params)
        categorizer = Categorizer(self.vocab, vocab_embeds, self.cf_feats)
        
        model_dir = data_dir+params['out_name']+'/'
        create_directory(model_dir)
        categorizer.set_datadir(model_dir)
        
        if params['prior_name'] != 'uniform':
            categorizer.add_prior(params['prior_name'], params['prior'])
            
        categorizer.run_categorization(slang_def_embeds, labels, models=[params['model']], prior=params['prior_name'], mode=mode, verbose=self.verbose)
    
    def get_results(self, fold_name='default', mode='train', params=None):
            
        if params is None:
            params = self.DEFAULT_PARAMS
        
        return np.load(self.data_dir + '/' + fold_name + '/'+params['out_name']+'/'+'l_'+params['model']+'_'+params['prior_name']+'_'+mode+'.npy')
