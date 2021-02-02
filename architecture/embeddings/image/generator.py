
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
import pickle
from torchvision import datasets, models, transforms
from torch import nn


class ImageEmbeddingGenerator():

    def __init__(self, model_name):
        if model_name == "vgg16":
            self.model = models.vgg16(pretrained=True)
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-3])
            self.extension = "vgg16_features"

    def generate_embeddings(self, iterator_list, outdir):
        self.model.eval()
        if not os.path.isdir(os.path.join(outdir, "img_embeddings")):
            os.mkdir(os.path.join(outdir, "img_embeddings"))
        for iterator in iterator_list:
            for batch in tqdm(iterator):
                with torch.no_grad():
                    features = self.model(batch["img"])

                for i, key in enumerate(batch["key"]):
                    np.save(os.path.join(outdir, "img_embeddings", f"{key}_{self.extension}.npy"), features[i])
