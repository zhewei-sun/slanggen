# Copyright (C) 2021 Zhewei Sun

import numpy as np
import pandas as pd

import scipy.spatial.distance as dist

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader
from sentence_transformers.readers import TripletReader
from sentence_transformers.evaluation import TripletEvaluator

from tqdm import trange

from .util import *
from .encoder import SBertEncoder

class SlangGenTrainer:
    
    MAX_NEIGHBOR = 300
    
    def __init__(self, dataset, word_encoder, out_dir='', verbose=False):
        
        self.out_dir = out_dir
        create_directory(out_dir)
        
        self.dataset = dataset
        
        self.word_encoder = word_encoder
        
        self.verbose = verbose
            
        conv_lens = []
        for i in range(dataset.V):
            word = dataset.vocab[i]
            conv_lens.append(len(dataset.conv_data[word].definitions))
        self.conv_lens = np.asarray(conv_lens)

        self.conv_acc = np.zeros(dataset.V, dtype=np.int32)

        for i in range(1,dataset.V):
            self.conv_acc[i] = self.conv_acc[i-1] + self.conv_lens[i-1]
            
        self.word_dist = self.preprocess_word_dist()
        np.save(out_dir+'/word_dist.npy', self.word_dist)
        
        self.sense_encoder = None
        self.se_model_name = "INVALID"

    def preprocess_slang_data(self, slang_ind, fold_name='default', skip_steps=[]):

        out_dir = self.out_dir + '/' + fold_name
        create_directory(out_dir)
        out_dir += '/'
        
        # Generate contrastive pairs for training
        if 'contrastive' not in skip_steps:
            if self.verbose:
                print("Generating contrative pairs...")
            contrastive_pairs_train, contrastive_pairs_dev = self.preprocess_contrastive(slang_ind)
            np.save(out_dir+'contrastive_train.npy', contrastive_pairs_train)
            np.save(out_dir+'contrastive_dev.npy', contrastive_pairs_dev)
            if self.verbose:
                print("Complete!")
                
    def load_preprocessed_data(self, fold_name='default', skip_steps=[]):
        
        out_dir = self.out_dir + '/' + fold_name + '/'
        
        preproc_data = {}
        
        if 'contrastive' not in skip_steps:
            preproc_data['cp_train'] = np.load(out_dir+'contrastive_train.npy', allow_pickle=True)
            preproc_data['cp_dev'] = np.load(out_dir+'contrastive_dev.npy', allow_pickle=True)
            
        return preproc_data
    
    def load_sense_encoder(self, model_name, model_path):
        
        if self.se_model_name == model_name:
            return self.sense_encoder
        
        self.sense_encoder = SBertEncoder(sbert_model_name=model_name, name=model_path)
        self.se_model_name = model_name
        
    
    def get_trained_embeddings(self, slang_ind, fold_name='default', model_path='SBERT_contrastive'):
        
        model_name = self.out_dir + '/' + fold_name + '/SBERT_data/' + model_path
        self.load_sense_encoder(model_name, model_path)
        
        return self.get_sense_embeddings(slang_ind, fold_name)
        
    def get_sense_embeddings(self, slang_ind, fold_name='default'):
                    
        if self.verbose:
            print("Encoding sense definitions...")
            
        out_dir = self.out_dir + '/' + fold_name + '/'
            
        sense_embeds = self.sense_encoder.encode_dataset(self.dataset, slang_ind)
        np.savez(out_dir+"sum_embed_"+self.sense_encoder.name+".npz", train=sense_embeds['train'], dev=sense_embeds['dev'], test=sense_embeds['test'], standard=sense_embeds['standard'])
        
        if self.verbose:
            print("Complete!")
            
        return sense_embeds
    
    def get_testtime_embeddings(self, slang_def_sents, fold_name='default', model_path='SBERT_contrastive'):
        
        model_name = self.out_dir + '/' + fold_name + '/SBERT_data/' + model_path
        self.load_sense_encoder(model_name, model_path)
        
        return self.sense_encoder.encode_sentences(slang_def_sents)
        
        
    def train_contrastive_model(self, slang_ind, params=None, fold_name='default'):
        
        if params is None:
            params = {'train_batch_size':16, 'num_epochs':4, 'triplet_margin':1, 'outpath':'SBERT_contrastive'}
        
        self.prep_contrastive_training(slang_ind, fold_name=fold_name)
        
        out_dir = self.out_dir + '/' + fold_name + '/SBERT_data/'

        triplet_reader = TripletReader(out_dir, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter=',', has_header=True)
        output_path = out_dir+params['outpath']
        
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        
        train_data = SentencesDataset(examples=triplet_reader.get_examples('contrastive_train.csv'), model=sbert_model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=params['train_batch_size'])
        train_loss = losses.TripletLoss(model=sbert_model, triplet_margin=params['triplet_margin'])

        dev_data = SentencesDataset(examples=triplet_reader.get_examples('contrastive_dev.csv'), model=sbert_model)
        dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=params['train_batch_size'])
        evaluator = TripletEvaluator(dev_dataloader)
        
        warmup_steps = int(len(train_data)*params['num_epochs']/params['train_batch_size']*0.1) #10% of train data

        # Train the model
        sbert_model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=params['num_epochs'],
                  evaluation_steps=len(dev_data),
                  warmup_steps=warmup_steps,
                  output_path=output_path)
    
    def prep_contrastive_training(self, slang_ind, fold_name='default'):
        
        if self.verbose:
            print("Generating triplet data for contrastive training...")
        
        out_dir = self.out_dir + '/' + fold_name + '/SBERT_data/'
        create_directory(out_dir)
        
        preproc_data = self.load_preprocessed_data(fold_name=fold_name)
        
        N_train, triplets = self.sample_triplets(preproc_data['cp_train'])
        N_dev, triplets_dev = self.sample_triplets(preproc_data['cp_dev'])
        
        np.save(out_dir+'triplets.npy', triplets)
        np.save(out_dir+'triplets_dev.npy', triplets_dev)
            
        slang_def_sents = []
        for i in range(self.dataset.N_total):
            slang_def_sents.append(' '.join(simple_preprocess(self.dataset.slang_data[i].def_sent)))

        conv_def_sents = []
        for i in range(self.dataset.V):
            word = self.dataset.vocab[i]
            for d in self.dataset.conv_data[word].definitions:
                conv_def_sents.append(' '.join(simple_preprocess(d['def'])))
        
        data_train = {'anchor':[slang_def_sents[slang_ind.train[triplets[i][0]]] for i in range(N_train)],\
                      'positive':[conv_def_sents[triplets[i][1]] for i in range(N_train)],\
                      'negative':[conv_def_sents[triplets[i][2]] for i in range(N_train)]}

        data_dev = {'anchor':[slang_def_sents[slang_ind.dev[triplets_dev[i][0]]] for i in range(N_dev)],\
                    'positive':[conv_def_sents[triplets_dev[i][1]] for i in range(N_dev)],\
                    'negative':[conv_def_sents[triplets_dev[i][2]] for i in range(N_dev)]}
        
        df_train = pd.DataFrame(data=data_train)
        df_dev = pd.DataFrame(data=data_dev)
        
        df_train.to_csv(out_dir+'contrastive_train.csv', index=False)
        df_dev.to_csv(out_dir+'contrastive_dev.csv', index=False)
        
        if self.verbose:
            print("Complete!")
        
    def sample_triplets(self, contrast_data):
    
        # Maximum number of positive pairs from the same positive definition
        MAX_PER_POSDEF = 1000
    
        triplets = []

        N_def = contrast_data.shape[0]

        for i in range(N_def):
            anchor = i
            if contrast_data[i]['negative'].shape[0] == 0:
                continue
            pre_pos = -100
            num_d = 0
            
            for positive in np.concatenate([contrast_data[i]['positive'], contrast_data[i]['neighbors']]):
                if positive != pre_pos+1:
                    num_d = MAX_PER_POSDEF
                pre_pos = positive
                if num_d > 0:
                    num_d -= 1

                    negative = np.random.choice(contrast_data[i]['negative'])
                    triplets.append(Triplet(anchor, positive, negative))

        N_triplets = len(triplets)

        if self.verbose:
            print("Sampled %d Triplets" % N_triplets)

        return N_triplets, np.asarray(triplets)
    
    def preprocess_word_dist(self):
    
        vocab_conv_embeds = np.zeros((self.dataset.V, self.word_encoder.E))

        for i in range(self.dataset.V):
            if self.dataset.vocab[i] in self.word_encoder.vocab:
                vocab_conv_embeds[i,:] = self.word_encoder.norm_embed(self.dataset.vocab[i])
            else:
                c_words = self.dataset.vocab[i].split(' ')
                count = 0
                if len(c_words) > 1:
                    embed = np.zeros(self.word_encoder.E)
                    for w in c_words:
                        if w in self.word_encoder.vocab:
                            embed = embed + self.word_encoder.norm_embed(w)
                            count += 1
                    if count > 0:
                        vocab_conv_embeds[i,:] = embed / float(count)

                if count == 0:
                    vocab_conv_embeds[i,:] = self.word_encoder.norm_embed('unk')

        return dist.squareform(dist.pdist(vocab_conv_embeds, metric='cosine'))

    def preprocess_contrastive(self, slang_ind):
        
        Neigh_pivot = int(np.ceil(self.dataset.V/5.0))
        N_neighbor = min(self.MAX_NEIGHBOR, self.dataset.V - Neigh_pivot)

        self.neighbors = np.zeros((self.dataset.V, N_neighbor), dtype=np.int32)
        self.neighbors_close = np.zeros((self.dataset.V, 5), dtype=np.int32)
        for i in range(self.dataset.V):
            self.neighbors[i,:] = np.argsort(self.word_dist[i,:])[max(Neigh_pivot, self.dataset.V-self.MAX_NEIGHBOR):]
            self.neighbors_close[i,:] = np.argsort(self.word_dist[i,:])[1:6]
            
        contrastive_pairs_train = self.compute_contrastive(slang_ind.train)
        contrastive_pairs_dev = self.compute_contrastive(slang_ind.dev)
        
        return contrastive_pairs_train, contrastive_pairs_dev
            
    def compute_contrastive(self, ind):
        
        def get_conv_definds(word_ind):
            return [self.conv_acc[word_ind]+j for j in range(self.conv_lens[word_ind])]
        
        contrast_data = np.empty(ind.shape[0], dtype=object)

        for i in trange(ind.shape[0]):
            word_ind = self.dataset.vocab_ids[ind[i]]
            contrast_data[i] = {}

            positives = [self.conv_acc[word_ind]+j for j in range(self.conv_lens[word_ind])]

            negatives = []
            conv_self = [d['def'] for d in self.dataset.conv_data[self.dataset.vocab[word_ind]].definitions]
            for far_word in self.neighbors[word_ind]:
                conv_defs = [d['def'] for d in self.dataset.conv_data[self.dataset.vocab[far_word]].definitions]
                for j in range(self.conv_lens[far_word]):
                    cand = self.conv_acc[far_word] + j
                    if not is_close_def(self.dataset.slang_data[ind[i]].def_sent, conv_defs[j], threshold=0.2):
                        has_close_cf_def = False
                        for self_def in conv_self:
                            if is_close_def(self_def, conv_defs[j], threshold=0.2):
                                has_close_cf_def = True
                                break
                        if not has_close_cf_def:
                            negatives.append(cand)            
            
            neigh_defs = []
            for close_word in self.neighbors_close[word_ind]:
                neigh_defs.extend([self.conv_acc[close_word]+j for j in range(self.conv_lens[close_word])])

            contrast_data[i]['positive'] = np.asarray(positives)
            contrast_data[i]['negative'] = np.asarray(negatives)
            contrast_data[i]['neighbors'] = np.asarray(neigh_defs)

        return contrast_data