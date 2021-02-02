
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses
from sentence_transformers.readers import InputExample
from sklearn.decomposition import PCA
from nltk.tokenize import RegexpTokenizer


class TextEmbeddingGenerator():

    def __init__(self,):
        pass

    def generate_sbert_embeddings(self, texts, model, n_components=None):
        embeddings = []
        dim = 0
        if not n_components:
            skip_pca = True
        elif n_components >= len(texts) or n_components >= 756:
            skip_pca = True
        else:
            skip_pca = False
        if not skip_pca:
            pca = PCA(n_components=n_components)

        for text in tqdm(texts):
            embeddings.append(model.encode(text))
        embeddings = np.stack(embeddings, axis=0)
        if not skip_pca:
            embeddings = pca.fit_transform(embeddings)

        embed_dict = {}
        for embedding, text in zip(embeddings, texts):
            embed_dict[text] = embedding
        dim = len(random.choice(list(embed_dict.values())))
        return embed_dict, dim

    def generate_phoc_embeddings(self, texts):
        pass

    def generate_bow_embeddings(self, texts):
        tokenizer = RegexpTokenizer(r'\w+')
        word_to_ix = {}
        dim = 0
        for sent in texts:
            tokens = tokenizer.tokenize(sent)
            for word in tokens:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

        embed_dict = {}
        for sent in tqdm(texts):
            tokens = tokenizer.tokenize(sent)
            embed_dict[sent] = self._make_bow_vector(tokens, word_to_ix)
        dim = len(random.choice(list(embed_dict.values())))
        return embed_dict, dim

    def _make_bow_vector(self, sentence, word_to_ix):
        vec = torch.zeros(len(word_to_ix))
        for word in sentence:
            vec[word_to_ix[word]] += 1
        return vec

    def finetune(self, train_examples, model, output_path=None, epochs=1):
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=24)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs,
                  warmup_steps=100, output_path=output_path)
        return model
