import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from nltk.tokenize import RegexpTokenizer
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import json
from sentence_transformers import SentenceTransformer
import math

# from models import RNN_ENCODER


class CUB200DataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size=24, im_size=256, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.imsize = im_size
        self.batch_size = batch_size
        self.image_transform = transforms.Compose([
            transforms.Resize(int(self.imsize * 76 / 64)),
            transforms.RandomCrop(self.imsize),
            transforms.RandomHorizontalFlip()])
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, self.imsize, self.imsize)
        self.num_workers = num_workers

    # def prepare_data(self):
    #     # download
    #     MNIST(self.data_dir, train=True, download=True)
    #     MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.cub200_test = CUB200Dataset(self.data_dir, transform=self.image_transform, split="test")
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.cub200_train = CUB200Dataset(self.data_dir, transform=self.image_transform, split="test")

    def train_dataloader(self):
        return DataLoader(self.cub200_train, batch_size=self.batch_size, drop_last=True,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cub200_test, batch_size=self.batch_size, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cub200_test, batch_size=self.batch_size, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True)


class CUB200Dataset(data.Dataset):
    def __init__(self, data_dir, transform=None, split='train', preprocess=False):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.data = []
        self.data_dir = data_dir
        if preprocess:
            self.preprocess_images()
            self.preprocesses_text()
        self.filenames, self.captions, self.embeddings = self.load_data(split)

    def load_data(self, split):
        filenames = self.load_filenames(self.data_dir, split)
        captions = {}
        embeddings = {}
        for filename in filenames:
            captions[filename] = self.load_captions(os.path.join(
                self.data_dir, "text", f"{filename}.txt"))
            embeddings[filename] = np.load(os.path.join(self.data_dir, "text", f"{filename}_distilroberta.npy"))
        return filenames, captions, embeddings

    def preprocess_images(self):
        bbox_dict = self.load_bbox()
        train_names = self.load_filenames(self.data_dir, 'train')
        test_names = self.load_filenames(self.data_dir, 'test')
        filenames = train_names + test_names
        if not os.path.isdir(os.path.join(self.data_dir, "processed_images")):
            os.mkdir(os.path.join(self.data_dir, "processed_images"))
        print("Preprocessing images")
        for filename in tqdm(filenames):
            bbox = bbox_dict[filename]
            img = Image.open(os.path.join(self.data_dir, "images", filename + ".jpg")).convert('RGB')
            width, height = img.size
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])

            bird_name = os.path.split(filename)[0]
            if not os.path.isdir(os.path.join(self.data_dir, "processed_images", bird_name)):
                os.mkdir(os.path.join(self.data_dir, "processed_images", bird_name))

            img.save(os.path.join(self.data_dir, "processed_images", filename + ".jpg"))

    def preprocesses_text(self):
        train_names = self.load_filenames(self.data_dir, 'train')
        test_names = self.load_filenames(self.data_dir, 'test')
        filenames = train_names + test_names
        text_encoder = SentenceTransformer('paraphrase-distilroberta-base-v1')
     #   text_encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        print("Generating embeddings")
        for filename in tqdm(filenames):
            captions = self.load_captions(os.path.join(self.data_dir, "text", filename + ".txt"))
            embeddings = text_encoder.encode(captions, convert_to_numpy=True)
            os.remove(os.path.join(self.data_dir, "text", f"{filename}_distilroberta.npz"))
            np.save(os.path.join(self.data_dir, "text", f"{filename}_distilroberta.npy"), embeddings)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
      #  print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, filename):
        with open(filename, "r") as f:
            captions = f.read().split('\n')
            captions = [x for x in captions if len(x) > 0]
            captions = [x.replace("\ufffd\ufffd", " ") for x in captions]
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                # print('tokens', tokens)
                if len(tokens) == 0:
                    print('cap', cap)
                    continue
        return captions

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
           # print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def load_image(self, filename):
        img_path = os.path.join(self.data_dir, "processed_images", filename + ".jpg")
        img = self.norm(Image.open(img_path).convert('RGB'))
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        #
        img = self.load_image(key)
        # random select a sentence
        rand_int = np.random.randint(0, len(self.captions[key]))
        caption = self.captions[key][rand_int]
        embedding = self.embeddings[key][rand_int]

        return img, embedding, caption

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    dataset = CUB200Dataset("/home/nino/Documents/Datasets/Birds", preprocess=True)
