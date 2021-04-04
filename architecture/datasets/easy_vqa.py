
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
from architecture.embeddings.text.generator import TextEmbeddingGenerator
from architecture.embeddings.image.generator import ImageEmbeddingGenerator
from sentence_transformers import SentenceTransformer
import pickle


class EasyVQADataModule(pl.LightningDataModule):

    def __init__(
            self, data_dir, batch_size=24, im_size=64, num_workers=4, pretrained_text=False,
            cnn_type=None, text_embed_type="sbert", norm=None, iterator="question"):
        super().__init__()
        self.data_dir = data_dir
        self.im_size = im_size
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.pretrained_text = pretrained_text
        self.cnn_type = cnn_type
#        self.pretrained_images = pretrained_images

        self.text_embed_type = text_embed_type
        self.iterator = iterator
        self.norm = norm

        self.question_embeddings, self.answer_embeddings = self.load_embeddings()
        if cnn_type:
            self.generate_image_embeddings()

    def load_embeddings(self):
        question_embeddings = None
        answer_embeddings = None
        if not os.path.exists(os.path.join(self.data_dir, f'{self.text_embed_type}_question_embeddings.pkl')):
            self.generate_text_embeds(self.text_embed_type)
        with open(os.path.join(self.data_dir, f'{self.text_embed_type}_question_embeddings.pkl'), "rb") as fIn:
            print(
                f"Loading question embeddings from {os.path.join(self.data_dir, f'{self.text_embed_type}_question_embeddings.pkl')}")
            question_embeddings = pickle.load(fIn)

        with open(os.path.join(self.data_dir, f'{self.text_embed_type}_answer_embeddings.pkl'), "rb") as fIn:
            answer_embeddings = pickle.load(fIn)

        return question_embeddings, answer_embeddings

    def setup(self, stage=None):

        image_transform = transforms.Compose([
            transforms.Resize(int(self.im_size))])

        self.easy_vqa_test = EasyVQADataset(
            self.data_dir, transform=image_transform, split="test", question_embeddings=self.question_embeddings,
            answer_embeddings=self.answer_embeddings, norm=self.norm, cnn_type=self.cnn_type, iterator=self.iterator)
        if stage == 'fit' or stage is None:
            self.easy_vqa_train = EasyVQADataset(
                self.data_dir, transform=image_transform, split="train", question_embeddings=self.question_embeddings,
                answer_embeddings=self.answer_embeddings, norm=self.norm, cnn_type=self.cnn_type,
                iterator=self.iterator)
            self.easy_vqa_val = EasyVQADataset(
                self.data_dir, transform=image_transform, split="val", question_embeddings=self.question_embeddings,
                answer_embeddings=self.answer_embeddings, norm=self.norm, cnn_type=self.cnn_type,
                iterator=self.iterator)

    def train_dataloader(self):
        return DataLoader(self.easy_vqa_train, batch_size=self.batch_size, drop_last=True,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.easy_vqa_test, batch_size=self.batch_size, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.easy_vqa_test, batch_size=self.batch_size, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True)

    def get_ef_dim(self, combined=False):
        if combined:
            return len(list(self.question_embeddings.values())[0]) + len(list(self.answer_embeddings.values())[0])
        else:
            return len(list(self.question_embeddings.values())[0])

    def get_answer_map(self):
        answers_file = os.path.join(self.data_dir, "answers.txt")
        with open(answers_file, 'r') as file:
            answers = dict((key, value.strip()) for key, value in enumerate(file))
        return answers

    def generate_text_embeds(self, type="sbert"):
        train_dataset = EasyVQADataset(self.data_dir, split="train")
        val_dataset = EasyVQADataset(self.data_dir, split="val")
        test_dataset = EasyVQADataset(self.data_dir, split="test")

        questions = train_dataset.get_questions() | val_dataset.get_questions() | test_dataset.get_questions()
        answers = train_dataset.get_answers() | val_dataset.get_answers() | test_dataset.get_answers()

        generator = TextEmbeddingGenerator()
        print(f"Generating {type} embeddings...")

       # n_components = len(self.get_answer_map()) - 1
        n_components = 512
        if type == "sbert":
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            question_embeddings, question_dim = generator.generate_sbert_embeddings(
                questions, model, n_components=n_components)
            answer_embeddings, answer_dim = generator.generate_sbert_embeddings(
                answers, model, n_components=n_components)

        elif type == "sbert_finetuned":
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = train_dataset.get_sentence_pairs()
            model = generator.finetune(train_examples, model, epochs=2)
            question_embeddings, question_dim = generator.generate_sbert_embeddings(
                questions, model, n_components=n_components)
            answer_embeddings, answer_dim = generator.generate_sbert_embeddings(
                answers, model, n_components=n_components)
        elif type == "bow":
            question_embeddings, question_dim = generator.generate_bow_embeddings(questions)
            answer_embeddings, answer_dim = generator.generate_bow_embeddings(answers)
        else:
            print(f"Unsupported embedding type: {type}")
            return

        print(f"{type} question embedding dim={question_dim}")
        print(f"{type} answer embedding dim={answer_dim}")
        with open(os.path.join(self.data_dir, f'{type}_question_embeddings.pkl'), "wb") as fOut:
            pickle.dump(question_embeddings, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.data_dir, f'{type}_answer_embeddings.pkl'), "wb") as fOut:
            pickle.dump(answer_embeddings, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    def generate_image_embeddings(self):
        if self.cnn_type == "frcnn":
            norm = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor()])
        else:
            norm = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        train_dataset = EasyVQADataset(self.data_dir, split="train", norm=norm, iterator="image")
        val_dataset = EasyVQADataset(self.data_dir, split="val", norm=norm, iterator="image")
        generator = ImageEmbeddingGenerator(self.data_dir, self.cnn_type)

        batch_size = 1

        train_outdir = os.path.join(self.data_dir, "train", 'embeddings', generator.extension)
        if not os.path.exists(train_outdir):
            os.makedirs(train_outdir)
        if len(os.listdir(train_outdir)) < len(train_dataset):
            print("Generating image embeddings for train dataset...")
            generator.generate_embeddings(DataLoader(
                train_dataset, batch_size=batch_size,
                num_workers=self.num_workers, pin_memory=True), train_outdir)

        val_outdir = os.path.join(self.data_dir, "val", 'embeddings', generator.extension)
        if not os.path.exists(val_outdir):
            os.makedirs(val_outdir)
        if len(os.listdir(val_outdir)) < len(val_dataset):
            print("Generating image embeddings for val dataset...")
            generator.generate_embeddings(DataLoader(
                train_dataset, batch_size=batch_size,
                num_workers=self.num_workers, pin_memory=True), val_outdir)


class EasyVQADataset(data.Dataset):

    def __init__(self, data_dir, transform=None, split='train', norm=None, question_embeddings=None,
                 answer_embeddings=None, cnn_type=None, iterator="question"):
        self.data_dir = data_dir
        self.cnn_type = cnn_type
        self.iterator = iterator
        self.split = split
        if norm:
            self.norm = norm
        else:
            # Norm to -1 1
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform = transform

        self.split_dir = os.path.join(data_dir, split)

        self.im_dir = os.path.join(self.split_dir, "images")

        self.question_embeddings = question_embeddings
        self.answer_embeddings = answer_embeddings
        self.questions, self.answers, self.images, self.imgToQA = self.load_data()

    def load_data(self):
        questions_file = os.path.join(self.split_dir, "questions.json")
        answers_file = os.path.join(self.data_dir, "answers.txt")
        with open(answers_file, 'r') as file:
            answers = dict((value.strip(), i) for i, value in enumerate(file))
        # answers = {"yes": 0, "no": 1, "circle": 2, "rectangle": 3, "triangle": 4, "red": 5,
        #            "green": 6, "blue": 7, "black": 8, "gray": 9, "teal": 10, "brown": 11, "yellow": 12}
        with open(questions_file, 'r') as file:
            questions = json.load(file)

            imgToQA = {}

            # q[0] question q[1] answer q[2] corresponding image id
            for q in questions:
                qa = (q[0], q[1])
                if q[2] not in imgToQA.keys():
                    imgToQA[q[2]] = []
                imgToQA[q[2]].append(qa)
            images = sorted(imgToQA.keys())
            return questions, answers, images, imgToQA

    def get_sentence_pairs(self):
        qa_pairs = []
        for qas in self.imgToQA.values():
            for qa in qas:
                qa_pairs.append(InputExample(texts=[qa[0], qa[1]]))
        return qa_pairs

    def get_questions(self):
        return set([q[0] for q in self.questions])

    def get_answers(self):
        return set([q[1] for q in self.questions])

    def load_image(self, image_id):
        img = Image.open(os.path.join(self.im_dir, f"{self.split}_{image_id}.png")).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return self.norm(img)

    def load_image_features(self, image_id):
        return np.load(os.path.join(
            self.data_dir, self.split, "embeddings", self.cnn_type,
            f"{self.split}_{image_id}.npy"))

    def __getitem__(self, index):
        if self.iterator == "question":
            question = self.questions[index][0]
            answer = self.questions[index][1]
            image_idx = self.questions[index][2]
        elif self.iterator == "image":
            image_idx = self.images[index]
            qa_list = self.imgToQA[image_idx]
            qa = random.choice(qa_list)
            question = qa[0]
            answer = qa[1]
        else:
            question = self.questions[index][0]
            answer = self.questions[index][1]
            image_idx = self.questions[index][2]

        img = self.load_image(image_idx)
        text = f'{question} {answer}'
        qa_embedding = 0
        q_embedding = 0
        if self.question_embeddings:
            q_embedding = self.question_embeddings[question]
            if self.answer_embeddings:
                qa_embedding = np.concatenate([self.question_embeddings[question], self.answer_embeddings[answer]])

        if self.cnn_type and self.split != "test":
            img_embedding = self.load_image_features(image_idx)
        else:
            img_embedding = 0

        return {'img_path': f"{self.split}_{image_idx}.png", "target": self.answers[answer], "img": img, "img_embedding": img_embedding, "question": question, "answer": answer, "text": text, "qa_embedding": qa_embedding, "q_embedding": q_embedding}

    def __len__(self):
        if self.iterator == "question":
            return len(self.questions)
        elif self.iterator == "image":
            return len(self.imgToQA.keys())
        else:
            return len(self.questions)


if __name__ == "__main__":
    data_dir = "/home/nino/Documents/Datasets/ExtEasyVQA/"
    datamodule = EasyVQADataModule(data_dir=data_dir, num_workers=1)
   # datamodule.generate_text_embeds(type="bow")
    # datamodule.generate_text_embeds(type="sbert")
    # datamodule.generate_text_embeds(type="sbert_finetuned")
   # datamodule.generate_image_embeddings()

    # datamodule = EasyVQADataModule(data_dir=data_dir, num_workers=1, pretrained_images=True, pretrained_text=True)
    # datamodule.setup()
    # for batch in datamodule.train_dataloader():
    #     print(1)
