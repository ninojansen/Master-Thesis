
import os
import pickle
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
import itertools


class TextEmbeddingGenerator():

    def __init__(self, ef_type=None, data_dir=None):
        if ef_type:
            if ef_type == "sbert_full":
                self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            elif ef_type == "sbert_reduced":
                self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
                with open(os.path.join(data_dir, "sbert_reduced_pca.pkl"), "rb") as fIn:
                    self.pca = pickle.load(fIn)
            elif ef_type == "phoc_reduced":
                with open(os.path.join(data_dir, "phoc_reduced_pca.pkl"), "rb") as fIn:
                    self.pca = pickle.load(fIn)
            elif ef_type == "bow":
                with open(os.path.join(data_dir, "bow_word_to_ix.pkl"), "rb") as fIn:
                    self.word_to_ix = pickle.load(fIn)
            self.ef_type = ef_type

    def process_batch(self, texts):
        tokenizer = RegexpTokenizer(r'\w+')
        if self.ef_type == "sbert_full" or self.ef_type == "sbert_reduced":
            embeddings = self.model.encode(texts)
            if self.ef_type == "sbert_reduced":
                embeddings = self.pca.transform(embeddings)
        elif self.ef_type == "phoc_full" or self.ef_type == "phoc_reduced":
            embeddings = []
            for text in texts:
                tokens = tokenizer.tokenize(text)
                phoc_words = []
                for word in tokens:
                    phoc_words.append(self._make_phoc_vector(word))
                embeddings.append(np.mean(np.vstack(phoc_words), axis=0, dtype=np.float32))
            embeddings = np.vstack(embeddings)
            if self.ef_type == "phoc_reduced":
                embeddings = self.pca.transform(embeddings)

        elif self.ef_type == "bow":
            embeddings = []
            for text in texts:
                tokens = tokenizer.tokenize(text)
                embeddings.append(self._make_bow_vector(tokens, self.word_to_ix))
            embeddings = np.vstack(embeddings)
        return torch.from_numpy(embeddings)

    def generate_sbert_embeddings(self, input_dict, model, reduce_features=False):
        dim = None
        pca = None
        embeddings = {}

        for name, texts in input_dict.items():
            embeddings[name] = []
            for text in tqdm(texts):

                embeddings[name].append(model.encode(text))

        if reduce_features:
            pca = PCA(n_components=0.98)
            pca.fit(np.vstack(list(embeddings.values())))

            for name, embs in embeddings.items():
                embeddings[name] = pca.transform(embs)

        for name, embs in embeddings.items():
            embed_dict = {}
            for embedding, text in zip(embeddings[name], input_dict[name]):
                embed_dict[text] = embedding
            embeddings[name] = embed_dict
            if not dim:
                dim = len(random.choice(list(embed_dict.values())))

        return embeddings, dim, pca

    def generate_phoc_embeddings(self, input_dict, reduce_features=False):
        tokenizer = RegexpTokenizer(r'\w+')
        dim = None
        pca = None
        embeddings = {}

        for name, texts in input_dict.items():
            embeddings[name] = []
            for text in tqdm(texts):
                tokens = tokenizer.tokenize(text)
                phoc_words = []
                for word in tokens:
                    phoc_words.append(self._make_phoc_vector(word))

                embeddings[name].append(np.mean(np.vstack(phoc_words), axis=0, dtype=np.float32))

        if reduce_features:
            pca = PCA(n_components=0.98)
            pca.fit(np.vstack(list(embeddings.values())))

            for name, embs in embeddings.items():
                embeddings[name] = pca.transform(embs)

        for name, embs in embeddings.items():
            embed_dict = {}
            for embedding, text in zip(embeddings[name], input_dict[name]):
                embed_dict[text] = embedding
            embeddings[name] = embed_dict
            if not dim:
                dim = len(random.choice(list(embed_dict.values())))

        return embeddings, dim, pca

    def generate_bow_embeddings(self, input_dict):
        tokenizer = RegexpTokenizer(r'\w+')
        word_to_ix = {"OOV": 0}

        dim = None
        embeddings = {}

        for sent_list in list(input_dict.values()):
            for sent in sent_list:
                tokens = tokenizer.tokenize(sent)
                for word in tokens:
                    word = word.lower()
                    if word not in word_to_ix:
                        word_to_ix[word] = len(word_to_ix)

        for name, texts in input_dict.items():
            embed_dict = {}
            for text in tqdm(texts):
                tokens = tokenizer.tokenize(text)
                embed_dict[text] = self._make_bow_vector(tokens, word_to_ix)
            embeddings[name] = embed_dict
            if not dim:
                dim = len(random.choice(list(embed_dict.values())))

        return embeddings, dim, word_to_ix

    def _make_phoc_vector(self, word):
        word = word.lower()
        hist_l1 = np.zeros(26)
        hist_l2_1 = np.zeros(26)
        hist_l2_2 = np.zeros(26)
        hist_l3_1 = np.zeros(26)
        hist_l3_2 = np.zeros(26)
        hist_l3_3 = np.zeros(26)
        hist_l3_4 = np.zeros(26)

        size = len(word)
        half_split = size // 2
        quart_split = size // 4
        # L1: full word
        for c in word:
            # 97 == ord("a")
            hist_l1[ord(c) - 97] += 1

        # L2: half word
        for c in word[:size // 2]:
            hist_l2_1[ord(c) - 97] += 1
        for c in word[size // 2:]:
            hist_l2_2[ord(c) - 97] += 1

        # L3 quarter split
        for c in word[:quart_split]:
            hist_l3_1[ord(c) - 97] += 1
        for c in word[quart_split:half_split]:
            hist_l3_2[ord(c) - 97] += 1
        for c in word[half_split:quart_split + half_split]:
            hist_l3_3[ord(c) - 97] += 1
        for c in word[quart_split + half_split:]:
            hist_l3_4[ord(c) - 97] += 1

        # Concatenate all vectors and normalize
        # embedding = np.concatenate((hist_l1, hist_l2_1, hist_l2_2, hist_l3_1,
        #                             hist_l3_2, hist_l3_3, hist_l3_4)) / (size * 3)
        # DONT NORMALIZE
        embedding = np.concatenate((hist_l1, hist_l2_1, hist_l2_2, hist_l3_1,
                                    hist_l3_2, hist_l3_3, hist_l3_4))
        return embedding

    def _make_bow_vector(self, sentence, word_to_ix):
        vec = torch.zeros(len(word_to_ix))
        for word in sentence:
            if word.lower() not in word_to_ix.keys():
                # Word is Out of vocabluary (OOV)
                vec[word_to_ix["OOV"]] += 1
            else:
                vec[word_to_ix[word.lower()]] += 1
        return vec

    def finetune(self, train_examples, model, output_path=None, epochs=1):
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=24)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs,
                  warmup_steps=100, output_path=output_path)
        return model
