
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import json
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sentence_transformers.readers import InputExample
from architecture.embeddings.generator import EmbeddingGenerator
from sentence_transformers import SentenceTransformer
import pickle


class EasyVQADataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size=24, im_size=64, num_workers=4, val_split=True):
        super().__init__()
        self.data_dir = data_dir
        self.im_size = im_size
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        image_transform = transforms.Compose([
            transforms.Resize(int(self.im_size * 76 / 64)),
            transforms.RandomCrop(self.im_size),
            transforms.RandomHorizontalFlip()])

        if stage == 'fit' or stage is None:
            easy_vqa = EasyVQADataset(self.data_dir, transform=image_transform, split="train")
            size = len(easy_vqa)
            train_size = math.ceil(size * 0.9)
            if self.val_split:
                self.easy_vqa_train, self.easy_vqa_val = random_split(
                    easy_vqa, [train_size, size - train_size],
                    generator=torch.Generator().manual_seed(42))
            else:
                self.easy_vqa_train = easy_vqa

        if stage == "test" or stage is None:
            self.easy_vqa_test = EasyVQADataset(self.data_dir, transform=image_transform, split="test")

    def train_dataloader(self):
        return DataLoader(self.easy_vqa_train, batch_size=self.batch_size, drop_last=True,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.easy_vqa_val, batch_size=self.batch_size, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.easy_vqa_test, batch_size=self.batch_size, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True)

    def pretrain_embeddings(self):
        train_dataset = EasyVQADataset(self.data_dir, split="train")
        test_dataset = EasyVQADataset(self.data_dir, split="test")

        texts = train_dataset.get_texts() | test_dataset.get_texts()
        generator = EmbeddingGenerator(n_components=64)
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        embeddings = generator.generate_embeddings(texts, model)
        with open(os.path.join(self.data_dir, 'sbert_embeddings.pkl'), "wb") as fOut:
            pickle.dump(embeddings, fOut, protocol=pickle.HIGHEST_PROTOCOL)

        # finetuned_embeddings = generator.generate_embeddings(
        #     self.get_texts(), generator.finetune(self.get_sentence_pairs(), model))
        # with open(os.path.join(self.data_dir, 'sbert_finetuned_embeddings.pkl'), "wb") as fOut:
        #     pickle.dump(finetuned_embeddings, fOut, protocol=pickle.HIGHEST_PROTOCOL)


class EasyVQADataset(data.Dataset):

    def __init__(self, data_dir, transform=None, split='train'):
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transform
        self.data_dir = data_dir

        self.split_dir = os.path.join(data_dir, split)

        self.im_dir = os.path.join(self.split_dir, "images")
        self.load_embeddings()
        self.image_ids, self.qa_dict = self.load_qa()

    def load_embeddings(self):
        with open(os.path.join(self.data_dir, 'sbert_embeddings.pkl'), "rb") as fIn:
            self.embeddings = pickle.load(fIn)

    def load_qa(self):
        questions_file = os.path.join(self.split_dir, "questions.json")
        with open(questions_file, 'r') as file:
            qs = json.load(file)
            qa_dict = {}
            # q[0] question q[1] answer q[2] corresponding image id
            for q in qs:
                qa = (q[0], q[1])
                if q[2] not in qa_dict.keys():
                    qa_dict[q[2]] = []
                qa_dict[q[2]].append(qa)

            return list(qa_dict.keys()), qa_dict

    def get_sentence_pairs(self):
        qa_pairs = []
        for qas in self.qa_dict.values():
            for qa in qas:
                qa_pairs.append(InputExample(texts=[qa[0], qa[1]]))
        return qa_pairs

    def get_texts(self):
        texts = set()
        for qas in self.qa_dict.values():
            for qa in qas:
                texts.add(qa[0])
                texts.add(qa[1])
        return texts

    def load_image(self, image_id):
        img = Image.open(os.path.join(self.im_dir, f"{image_id}.png")).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return self.norm(img)

    def __getitem__(self, index):

        qa_list = self.qa_dict[self.image_ids[index]]
        qa = random.choice(qa_list)
        question = qa[0]
        answer = qa[1]

        img = self.load_image(self.image_ids[index])

        embedding = np.concatenate([self.embeddings[question], self.embeddings[answer]])
        combined = f'{question} {answer}'
        return img, combined, embedding

    def __len__(self):
        return len(self.image_ids)


if __name__ == "__main__":
    data_dir = "/home/nino/Documents/Datasets/EasyVQA/data"
    datamodule = EasyVQADataModule(data_dir=data_dir)
    datamodule.pretrain_embeddings()
