import os
import pickle
import sys
from collections import defaultdict
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
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
from architecture.embeddings.rnn import RNN_ENCODER
# from models import RNN_ENCODER


class CUB200DataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size=24, embedding_type="RNN", im_size=256, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.imsize = im_size
        self.batch_size = batch_size
        self.embedding_type = embedding_type
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
        self.cub200_test = CUB200Dataset(self.data_dir, embedding_type=self.embedding_type,
                                         transform=self.image_transform, split="test")
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.cub200_train = CUB200Dataset(self.data_dir, embedding_type=self.embedding_type,
                                              transform=self.image_transform, split="test")

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
    def __init__(self, data_dir, transform=None, split='train', embedding_type="RNN", preprocess=False):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.data = []
        self.data_dir = data_dir
        self.embedding_type = embedding_type
        self.captions_per_image = 10
        if preprocess:
            self.ixtoword, self.wordtoix, self.n_words = self.build_dictionary()
        else:
            self.filenames, self.captions, self.embeddings = self.load_data(split)

    def load_data(self, split):
        filenames = self.load_filenames(self.data_dir, split)
        captions = {}
        embeddings = {}
        for filename in filenames:
            captions[filename] = self.load_captions(os.path.join(
                self.data_dir, "text", f"{filename}.txt"))
            if self.embedding_type == "RNN":
                embeddings[filename] = np.load(os.path.join(self.data_dir, "text", f"{filename}_rnn.npy"))
            elif self.embedding_type == "BERT":
                embeddings[filename] = np.load(os.path.join(self.data_dir, "text", f"{filename}_distilroberta.npy"))
        return filenames, captions, embeddings

    def build_dictionary(self):
        word_counts = defaultdict(float)
        train_names = self.load_filenames(self.data_dir, 'train')
        test_names = self.load_filenames(self.data_dir, 'test')
        filenames = train_names + test_names

        captions = []
        for filename in filenames:
            captions += self.load_captions(os.path.join(
                self.data_dir, "text", f"{filename}.txt"), join=False)
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        return [ixtoword, wordtoix, len(ixtoword)]

    def captions_toix(self, captions):
        # a list of indices for a sentence
        words_num = 18
        res_captions = []
        len_captions = []
        for cap in captions:
            rev = []
            for w in cap:
                if w in self.wordtoix:
                    rev.append(self.wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            sent_caption = np.asarray(rev).astype('int64')
            if (sent_caption == 0).sum() > 0:
                print('ERROR: do not need END (0) token', sent_caption)
            num_words = len(sent_caption)
            # pad with 0s (i.e., '<end>')
            x = np.zeros((words_num), dtype='int64')
            x_len = num_words
            if num_words <= words_num:
                x[:num_words] = sent_caption
            else:
                ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:words_num]
                ix = np.sort(ix)
                x[:] = sent_caption[ix]
                x_len = words_num
            res_captions.append(x)
            len_captions.append(x_len)
        return np.stack(res_captions), np.stack(len_captions)

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

    def preprocesses_text(self, encoder, type="BERT"):
        train_names = self.load_filenames(self.data_dir, 'train')
        test_names = self.load_filenames(self.data_dir, 'test')
        filenames = train_names + test_names

     #   text_encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        print(f"Generating {type} embeddings")
        for filename in tqdm(filenames):
            if type == "BERT":
                captions = self.load_captions(os.path.join(self.data_dir, "text", filename + ".txt"))
                embeddings = encoder.encode(captions, convert_to_numpy=True)
                np.save(os.path.join(self.data_dir, "text", f"{filename}_distilroberta.npy"), embeddings)
            elif type == "RNN":
                captions = self.load_captions(os.path.join(self.data_dir, "text", filename + ".txt"), join=False)
                hidden = text_encoder.init_hidden(len(captions))
                captions, cap_lens = self.captions_toix(captions)
                captions = torch.from_numpy(captions).cuda()
                with torch.no_grad():
                    _, embeddings = text_encoder(captions, cap_lens, hidden)
                    embeddings = embeddings.detach().cpu()
                    np.save(os.path.join(self.data_dir, "text", f"{filename}_rnn.npy"), embeddings)

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

    def load_captions(self, filename, join=True):
        all_captions = []
        with open(filename, "r") as f:
            captions = f.read().split('\n')
            cnt = 0
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

                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                all_captions.append(tokens_new)
                cnt += 1
                if cnt == self.captions_per_image:
                    break

            if cnt < self.captions_per_image:
                print('ERROR: the captions for %s less than %d'
                      % (filename, cnt))
        if join:
            return [" ".join(x) for x in all_captions]
        else:
            return all_captions

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
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        img = self.norm(img)

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
    dataset = CUB200Dataset("/home/nino/Documents/Datasets/CUB200", preprocess=True)

    # Processing image bounding boxes
#    dataset.preprocess_images()

    # BERT embeddings
    text_encoder = SentenceTransformer('paraphrase-distilroberta-base-v1')
    dataset.preprocesses_text(text_encoder, type="BERT")

    # RNN embeddings
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=256)
    state_dict = torch.load("/home/nino/Documents/Datasets/CUB200/encoders/text_encoder200.pth",
                            map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()
    dataset.preprocesses_text(text_encoder, type="RNN")
