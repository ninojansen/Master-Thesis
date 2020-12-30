import json
import math
import os
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
                    easy_vqa, [len(easy_vqa), size - train_size],
                    generator=torch.Generator().manual_seed(1))
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


class EasyVQADataset(data.Dataset):

    def __init__(self, data_dir, transform=None, split='train'):
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transform
        self.data_dir = data_dir

        self.split_dir = os.path.join(data_dir, split)

        self.im_dir = os.path.join(self.split_dir, "images")
        self.image_ids, self.qa_dict = self.load_qa()

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

    def load_image(self, image_id):
        img = self.norm(Image.open(os.path.join(self.im_dir, f"{image_id}.png")).convert('RGB'))
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, index):

        qa_list = self.qa_dict[self.image_ids[index]]
        qa = random.choice(qa_list)
        question = qa[0]
        answer = qa[1]

        img = self.load_image(self.image_ids[index])

        combined = f"{question} {answer}."
        return img, question, answer, combined

    def __len__(self):
        return len(self.image_ids)
