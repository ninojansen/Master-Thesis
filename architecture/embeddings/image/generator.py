
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
import shutil


class ImageEmbeddingGenerator():

    def __init__(self, model_name):
        if model_name == "vgg16":
            self.model = models.vgg16(pretrained=True)
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-3])
            self.extension = "vgg16"

    def generate_embeddings(self, dataloader, outdir):
        self.model.eval()
        outdir = os.path.join(outdir, 'embeddings', self.extension)
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)
        for batch in tqdm(dataloader):
            with torch.no_grad():
                features = self.model(batch["img"])
            for i, path in enumerate(batch["img_path"]):
                np.save(os.path.join(outdir, path.replace(".png", f"_{self.extension}.npy")), features[i])
