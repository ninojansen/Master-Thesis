
import os
import sys

from pytorch_lightning.core import datamodule
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
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
from tqdm import tqdm
from sentence_transformers.readers import InputExample
from architecture.embeddings.text.generator import TextEmbeddingGenerator
from sentence_transformers import SentenceTransformer
import pickle


class VQA2DataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size=24, im_size=256, num_workers=4, text_embed_type="sbert"):
        super().__init__()
        self.data_dir = data_dir
        self.im_size = im_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_embed_type = text_embed_type

        self.question_embeddings, self.answer_embeddings = self.load_embeddings()

    def load_embeddings(self):
        question_embeddings = None
        answer_embeddings = None
        if os.path.exists(os.path.join(self.data_dir, f'{self.text_embed_type}_question_embeddings.pkl')):
            with open(os.path.join(self.data_dir, f'{self.text_embed_type}_question_embeddings.pkl'), "rb") as fIn:
                question_embeddings = pickle.load(fIn)
        else:
            print(f"{self.text_embed_type} question embeddings do not exist at file location: {os.path.join(self.data_dir, f'{self.text_embed_type}_question_embeddings.pkl')}")
        if os.path.exists(os.path.join(self.data_dir, f'{self.text_embed_type}_answer_embeddings.pkl')):
            with open(os.path.join(self.data_dir, f'{self.text_embed_type}_answer_embeddings.pkl'), "rb") as fIn:
                answer_embeddings = pickle.load(fIn)
        else:
            print(f"{self.text_embed_type} answer embeddings do not exist at file location: {os.path.join(self.data_dir, f'{self.text_embed_type}_answer_embeddings.pkl')}")

        return question_embeddings, answer_embeddings

    def setup(self, stage=None):
        image_transform = transforms.Compose([
            transforms.Resize(int(self.im_size * 76 / 64)),
            transforms.RandomCrop(self.im_size),
            transforms.RandomHorizontalFlip()])

        if stage == 'fit' or stage is None:
            self.vqa2_train = VQA2Dataset(self.data_dir, transform=image_transform, split="train")
            self.vqa2_val = VQA2Dataset(self.data_dir, transform=image_transform, split="val")
        if stage == "test" or stage is None:
            # Test split has no answers  so using validation split instead
            self.vqa2_test = VQA2Dataset(self.data_dir, transform=image_transform, split="val",)
           # self.vqa2_test = VQA2Dataset(self.data_dir, transform=image_transform, split="test",)

    def train_dataloader(self):
        return DataLoader(self.vqa2_train, batch_size=self.batch_size, drop_last=True,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.vqa2_val, batch_size=self.batch_size, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.vqa2_test, batch_size=self.batch_size, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True)

    def generate_text_embeds(self, type="sbert"):
        train_dataset = VQA2Dataset(self.data_dir, split="train")
        val_dataset = VQA2Dataset(self.data_dir, split="val")

        # TODO implement these functions
        questions = train_dataset.get_questions() | val_dataset.get_questions()
        answers = train_dataset.get_answers() | val_dataset.get_answers()

        generator = TextEmbeddingGenerator()
        print(f"Generating {type} embeddings...")
        if type == "sbert":
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            question_embeddings, question_dim = generator.generate_sbert_embeddings(questions, model, n_components=256)
            answer_embeddings, answer_dim = generator.generate_sbert_embeddings(answers, model, n_components=256)

        elif type == "sbert_finetuned":
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = train_dataset.get_sentence_pairs()
            model = generator.finetune(train_examples, model, epochs=2)
            question_embeddings, question_dim = generator.generate_sbert_embeddings(questions, model, n_components=256)
            answer_embeddings, answer_dim = generator.generate_sbert_embeddings(answers, model, n_components=256)

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


class VQA2Dataset(data.Dataset):
    def __init__(self, data_dir, transform=None, split="train", answer_embeddings=None, question_embeddings=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        annotations_file = None
        self.split = split
        self.answer_embeddings = answer_embeddings
        self.question_embeddings = question_embeddings
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
            qa = {ann["question_id"]: [] for ann in annotations}
            qqa = {ann["question_id"]: [] for ann in annotations}
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

        file_name = f'COCO_{self.type}_{str(image_id).zfill(12)}.jpg'
        img = Image.open(os.path.join(self.images_folder, file_name)).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return self.norm(img)

    def get_questions(self):
        questions = set()
        for q in self.questions:
            questions.add(q["question"])
        return questions

    def get_answers(self):
        answers = set()
        for q in self.questions:
            possible_answers = self.qa[q["question_id"]]
            for answer in possible_answers["answers"]:
                answers.add(answer['answer'])
        return answers

    def get_sentence_pairs(self):
        qa_pairs = []
        for q in self.questions:
            possible_answers = self.qa[q["question_id"]]
            unique_answers = []
            for answer in possible_answers["answers"]:
                if answer["answer"] not in unique_answers:
                    unique_answers.append(answer["answer"])
                    qa_pairs.append(InputExample(texts=[q["question"], answer["answer"]]))

        return qa_pairs

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
        a = random.choice(a["answers"])
        answer = a["answer"]

        question = q["question"]
        text = f'{question} {answer}'
        qa_embedding = 0
        q_embedding = 0
        if self.question_embeddings:
            q_embedding = self.question_embeddings[question]
            if self.answer_embeddings:
                qa_embedding = np.concatenate([self.question_embeddings[question], self.answer_embeddings[answer]])
        return {'key': q["image_id"], "target": 0, "img": img, "img_embedding": 0, "question": question, "answer": answer, "text": text, "qa_embedding": qa_embedding, "q_embedding": q_embedding}

    def __len__(self):
        return len(self.questions)


if __name__ == "__main__":
    data_dir = "/home/nino/Documents/Datasets/VQA"
    datamodule = VQA2DataModule(data_dir=data_dir)
    datamodule.generate_text_embeds(type="bow")
    datamodule.generate_text_embeds(type="sbert")
    # datamodule.generate_text_embeds(type="sbert_finetuned")

    datamodule = VQA2DataModule(data_dir=data_dir, num_workers=1, pretrained_text=True)
    datamodule.setup()
    for batch in datamodule.train_dataloader():
        print(1)
