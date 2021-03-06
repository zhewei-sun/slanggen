# A Computational Framework for Slang Generation - Code and Data Repository

#### By: [Zhewei Sun](http://www.cs.toronto.edu/~zheweisun/)

This is the github repository for the TACL paper "A Computation Framework for Slang Generation".

## Urban Dictionary (UD) Dataset for Slang NLP

The dataset is a curated subset of the open source Urban Dictionary subset made available via Kaggle:

https://www.kaggle.com/therohk/urban-dictionary-words-dataset

Please see our paper for details regarding how the entries were selected and processed.

You can find the dataset under the /UD_Dataset directory.

The .csv files contain the data in raw text format (UD_Data_full.csv) and are split into training (UD_Data_train.csv) and testing (UD_Data_test.csv) partitions.

UD_Dataset.npy contains a python friendly format of the dataset and also includes sample trained contrastive embeddings based on both fastText and SBERT along with the baseline variants without contrastive training. See the accompanied IPython notebook (UD_Dataset.ipynb) for an usage example.

## Training Contrastive Sense Embeddings

Code for training contrastive embeddings coming soon. Stay Tuned!