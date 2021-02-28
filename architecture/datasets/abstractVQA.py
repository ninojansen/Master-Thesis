
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
from architecture.embeddings.image.generator import ImageEmbeddingGenerator
from sentence_transformers import SentenceTransformer
import pickle


class AbstractVQADataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size=24, im_size=256, num_workers=4, text_embed_type="sbert"):
        super().__init__()
        self.data_dir = data_dir
        self.im_size = im_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_embed_type = text_embed_type
        if not os.path.exists(os.path.join(self.data_dir, "answers.txt")):
            self.preprocess()
            self.generate_answers_file()
        self.question_embeddings, self.answer_embeddings = self.load_embeddings()

    def preprocess(self):
        print("Preparing data...")
        train_anno = json.load(open(os.path.join(self.data_dir, "annotations", "train_annotations.json"), 'r'))
        val_anno = json.load(open(os.path.join(self.data_dir, "annotations", "val_annotations.json"), 'r'))

        train_mc = json.load(open(os.path.join(self.data_dir, "questions", "mc_train_questions.json"), 'r'))
        val_mc = json.load(open(os.path.join(self.data_dir, "questions", "mc_val_questions.json"), 'r'))
        test_mc = json.load(open(os.path.join(self.data_dir, "questions", "mc_test_questions.json"), 'r'))

        train_oq = json.load(open(os.path.join(self.data_dir, "questions", "open_train_questions.json"), 'r'))
        val_oq = json.load(open(os.path.join(self.data_dir, "questions", "open_val_questions.json"), 'r'))
        test_oq = json.load(open(os.path.join(self.data_dir, "questions", "open_test_questions.json"), 'r'))

        train = []
        val = []
        test = []

        for i in range(len(train_anno['annotations'])):
            question_id = train_anno['annotations'][i]['question_id']
            image_path = f"abstract_v002_train2015_{train_anno['annotations'][i]['image_id']:012d}.png"
            question = train_oq['questions'][i]['question']
            ans = train_anno['annotations'][i]['multiple_choice_answer']
            train.append({'ques_id': question_id, 'img_path': image_path,
                          'question': question, 'ans': ans})
        for i in range(len(val_anno['annotations'])):
            question_id = val_anno['annotations'][i]['question_id']
            image_path = f"abstract_v002_val2015_{val_anno['annotations'][i]['image_id']:012d}.png"
            question = val_oq['questions'][i]['question']
            ans = val_anno['annotations'][i]['multiple_choice_answer']
            val.append({'ques_id': question_id, 'img_path': image_path,
                        'question': question, 'ans': ans})

        for i in range(len(test_oq['questions'])):
            question_id = test_oq['questions'][i]['question_id']
            question = test_oq['questions'][i]['question']
            image_path = f"abstract_v002_test2015_{test_oq['questions'][i]['image_id']:012d}.png"
            test.append({"ques_id": question_id, 'img_path': image_path, 'question': question})

        json.dump(train, open(os.path.join(self.data_dir, 'vqa_train.json'), 'w'))
        json.dump(val, open(os.path.join(self.data_dir, 'vqa_val.json'), 'w'))
        json.dump(test, open(os.path.join(self.data_dir, 'vqa_test.json'), 'w'))
        return

    def load_embeddings(self):
        question_embeddings = None
        answer_embeddings = None
        if not os.path.exists(os.path.join(self.data_dir, f'{self.text_embed_type}_question_embeddings.pkl')):
            self.generate_text_embeds(self.text_embed_type)
        with open(os.path.join(self.data_dir, f'{self.text_embed_type}_question_embeddings.pkl'), "rb") as fIn:
            question_embeddings = pickle.load(fIn)

        with open(os.path.join(self.data_dir, f'{self.text_embed_type}_answer_embeddings.pkl'), "rb") as fIn:
            answer_embeddings = pickle.load(fIn)

        return question_embeddings, answer_embeddings

    def setup(self, stage=None):
        image_transform = transforms.Compose([
            transforms.Resize(int(self.im_size * 76 / 64)),
            transforms.RandomCrop(self.im_size),
            transforms.RandomHorizontalFlip()])

        if stage == 'fit' or stage is None:
            self.vqa2_train = AbstractVQADataset(
                self.data_dir, transform=image_transform, split="train", question_embeddings=self.question_embeddings,
                answer_embeddings=self.answer_embeddings)
            self.vqa2_val = AbstractVQADataset(
                self.data_dir, transform=image_transform, split="val", question_embeddings=self.question_embeddings,
                answer_embeddings=self.answer_embeddings)
        if stage == "test" or stage is None:
            self.vqa2_test = AbstractVQADataset(
                self.data_dir, transform=image_transform, split="test", question_embeddings=self.question_embeddings,
                answer_embeddings=self.answer_embeddings)

    def train_dataloader(self):
        return DataLoader(self.vqa2_train, batch_size=self.batch_size, drop_last=True,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.vqa2_val, batch_size=self.batch_size, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.vqa2_test, batch_size=self.batch_size, drop_last=True,
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

    def generate_answers_file(self):
        train_dataset = AbstractVQADataset(self.data_dir, split="train")
        val_dataset = AbstractVQADataset(self.data_dir, split="val")
        answers = train_dataset.get_answers() | val_dataset.get_answers()
        with open(os.path.join(self.data_dir, 'answers.txt'), 'w') as file:
            for answer in answers:
                file.write(f'{answer}\n')

    def generate_text_embeds(self, type="sbert"):
        train_dataset = AbstractVQADataset(self.data_dir, split="train")
        val_dataset = AbstractVQADataset(self.data_dir, split="val")

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

    def generate_image_embeddings(self):
        image_transform = transforms.Compose([
            transforms.Resize(224)])
        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        train_dataset = AbstractVQADataset(self.data_dir, split="train", norm=norm, transform=image_transform)
        val_dataset = AbstractVQADataset(self.data_dir, split="val", norm=norm, transform=image_transform)
        test_dataset = AbstractVQADataset(self.data_dir, split="test", norm=norm, transform=image_transform)

        generator = ImageEmbeddingGenerator("vgg16")

        batch_size = 64
        generator.generate_embeddings(DataLoader(
            train_dataset, batch_size=batch_size,
            num_workers=self.num_workers, pin_memory=True), os.path.join(self.data_dir, 'images', 'train'))

        generator.generate_embeddings(DataLoader(
            val_dataset, batch_size=batch_size,
            num_workers=self.num_workers, pin_memory=True), os.path.join(self.data_dir, 'images', 'val'))

        generator.generate_embeddings(DataLoader(
            test_dataset, batch_size=batch_size,
            num_workers=self.num_workers, pin_memory=True), os.path.join(self.data_dir, 'images', 'test'))


class AbstractVQADataset(data.Dataset):
    def __init__(self, data_dir, transform=None, split="train", norm=None, answer_embeddings=None,
                 question_embeddings=None):
        self.transform = transform
        if norm:
            self.norm = norm
        else:
            # Norm to -1 1
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.split = split
        self.answer_embeddings = answer_embeddings
        self.question_embeddings = question_embeddings

        self.images_folder = os.path.join(data_dir, "images", split)
        self.data = json.load(open(os.path.join(data_dir, f"vqa_{split}.json"), 'r'))

        if os.path.exists(os.path.join(data_dir, "answers.txt")):
            with open(os.path.join(data_dir, "answers.txt"), 'r') as file:
                self.answers = dict((value.strip(), i) for i, value in enumerate(file))

    def load_image(self, file_name):
        img = Image.open(os.path.join(self.images_folder, file_name)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return self.norm(img)

    def get_questions(self):
        questions = set()
        for q in self.data:
            questions.add(q["question"])
        return questions

    def get_answers(self):
        answers = set()
        for q in self.data:
            answers.add(q["ans"])
        return answers

    def get_sentence_pairs(self):
        qa_pairs = []
        for q in self.data:
            qa_pairs.append(InputExample(texts=[q["question"], q["ans"]]))

        return qa_pairs

    def __getitem__(self, index):
        # Select the question
        data_dict = self.data[index]
        # Load the corresponding images
        question = data_dict["question"]
        if self.split == "test":
            ans = 0
            target = 0
        else:
            ans = data_dict['ans']
            target = self.answers[ans]
        img = self.load_image(data_dict["img_path"])

        text = f'{question} {ans}'

        qa_embedding = 0
        q_embedding = 0
        if self.question_embeddings:
            q_embedding = self.question_embeddings[question]
            if self.answer_embeddings:
                qa_embedding = np.concatenate([self.question_embeddings[question], self.answer_embeddings[ans]])
        return {'key': data_dict["ques_id"], "target": target, "img": img, "img_path": data_dict["img_path"], "img_embedding": 0, "question": question, "answer": ans, "text": text, "qa_embedding": qa_embedding, "q_embedding": q_embedding}

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_dir = "/home/nino/Documents/Datasets/AbstractVQA"
    datamodule = AbstractVQADataModule(data_dir=data_dir, im_size=64,)
    datamodule.generate_image_embeddings()
    # datamodule.generate_text_embeds(type="bow")
    # datamodule.generate_text_embeds(type="sbert")
    # # datamodule.generate_text_embeds(type="sbert_finetuned")

    # datamodule = VQA2DataModule(data_dir=data_dir, num_workers=1, pretrained_text=True)
    datamodule.setup(stage="fit")
    for batch in datamodule.train_dataloader():
        break
    datamodule.setup(stage="test")
    for batch in datamodule.train_dataloader():
        break
