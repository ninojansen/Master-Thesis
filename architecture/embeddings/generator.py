
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
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


class EmbeddingGenerator():

    def __init__(self, n_components=256):
        self.n_components = n_components

    def generate_embeddings(self, texts, model):
        embeddings = []
        pca = PCA(n_components=self.n_components)
        print("Generating embeddings...")
        for text in tqdm(texts):
            embeddings.append(model.encode(text))
        embeddings = np.stack(embeddings, axis=0)
        embeddings = pca.fit_transform(embeddings)

        embed_dict = {}
        for embedding, text in zip(embeddings, texts):
            embed_dict[text] = embedding
        return embed_dict

    def finetune(self, train_examples, model, output_path=None, epochs=1):
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=24)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs,
                  warmup_steps=100, output_path=output_path)
        return model
