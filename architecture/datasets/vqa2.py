import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import json
import matplotlib.pyplot as plt


class VQA2(data.Dataset):
    def __init__(self, data_dir, transform=None, split="train"):
        self.transform = transform

        annotations_file = None
        if split == "val":
            self.type = "val2014"
            self.images_folder = os.path.join(data_dir, "images/mscoco/val2014")

            questions_file = os.path.join(data_dir, "questions/v2_OpenEnded_mscoco_val2014_questions.json")
            annotations_file = os.path.join(data_dir, "annotations/v2_mscoco_val2014_annotations.json")
        elif split == "test":
            self.type = "test2015"
            self.images_folder = os.path.join(data_dir, "images/mscoco/test2015")

            questions_file = os.path.join(
                data_dir, "questions/v2_OpenEnded_mscoco_test2015_questions.json")
        else:
            self.type = "train2014"
            self.images_folder = os.path.join(data_dir, "images/mscoco/train2014")

            questions_file = os.path.join(
                data_dir, "questions/v2_OpenEnded_mscoco_train2014_questions.json")
            annotations_file = os.path.join(data_dir, "annotations/v2_mscoco_train2014_annotations.json")

       # self.annotations = edict(json.load(open(annotations_file, 'r')))
        self.load_data(questions_file, annotations_file)

    def load_data(self, questions_file, annotations_file):
        print('Loading QA data...')
        self.questions = json.load(open(questions_file, 'r'))["questions"]
        if annotations_file is not None:
            annotations = json.load(open(annotations_file, 'r'))["annotations"]

            imgToQA = {ann["image_id"]: [] for ann in annotations}
            qa = {ann["question_id"]:       [] for ann in annotations}
            qqa = {ann["question_id"]:       [] for ann in annotations}
            for ann in annotations:
                imgToQA[ann['image_id']] += [ann]
                qa[ann['question_id']] = ann
            for ques in self.questions:
                qqa[ques['question_id']] = ques

            # create class members
            self.qa = qa
            self.qqa = qqa
            self.imgToQA = imgToQA
        else:
            print("Test split so skipping annotations.")
        print('QA Data loaded!')

    def load_image(self, image_id):
        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        file_name = f'COCO_{self.type}_{str(image_id).zfill(12)}.jpg'
        img = norm(Image.open(os.path.join(self.images_folder, file_name)).convert('RGB'))
        if self.transform:
            img = self.transform(img)

        return img

    def __getitem__(self, index):
        # Select the question
        q = self.questions[index]
        # Load the corresponding images
        img = self.load_image(q["image_id"])

        # Answers are not provided for the test set
        if self.type == "test2015":
            return img, q["question"]

        a = self.qa[q["question_id"]]
        # Randomly sample an answer from the list of viable answers
        answer = random.choice(a["answers"])

        return q["question"], answer["answer"], img

    def __len__(self):
        return len(self.questions)


image_transform = transforms.Compose([
    transforms.Resize(int(256 * 76 / 64)),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip()])

dataset = VQA2(data_dir="/home/nino/Documents/Datasets/VQA", transform=image_transform, split="val")
loader = DataLoader(dataset, batch_size=1, drop_last=True,
                    shuffle=True, num_workers=1, pin_memory=True)

for batch in loader:
    q, a, i = batch
    print(q, a)
