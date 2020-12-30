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
from misc.config import cfg

if "sentence_transformers" in sys.modules:
    from sentence_transformers import SentenceTransformer

import math

from models import RNN_ENCODER


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    ret.append(normalize(img))
    # if cfg.GAN.B_DCGAN:
    '''
    for i in range(cfg.TREE.BRANCH_NUM):
        # print(imsize[i])
        re_img = transforms.Resize(imsize[i])(img)
        ret.append(normalize(re_img))
    '''

    return ret


class CUB200DataModule(pl.LightningDataModule):

    def __init__(self, data_dir,  batch_size=24, im_size=256, num_workers=4):
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
        self.cub200_test = CUB200Dataset(self.data_dir, "test", self.imsize,
                                         transform=self.image_transform, encoder_type=cfg.TEXT.ENCODER)
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.cub200_train = CUB200Dataset(self.data_dir, "train", self.imsize,
                                              transform=self.image_transform, encoder_type=cfg.TEXT.ENCODER)
            # size = len(cub200_full)
            # train_size = math.ceil(size * 0.9)

            # self.cub200_train, self.cub200_val = random_split(
            #     cub200_full, [train_size, size - train_size],
            #     generator=torch.Generator().manual_seed(42))

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
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None, encoder_type="RNN"):
        self.encoder_type = encoder_type
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.raw_strings = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
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

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
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
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
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

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        pretrained_filepath = os.path.join(data_dir, f'captions_{self.encoder_type}.pickle')

        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')

        # No pretrained embeddings
        if not os.path.isfile(pretrained_filepath):
            print(f"Found no pretrained embbedings in {pretrained_filepath}.")
            captions_filepath = os.path.join(data_dir, 'captions.pickle')
            if not os.path.isfile(captions_filepath):
                print(f"Found no captions file in {captions_filepath}")
                print("Generating new captions file...")
                # Generate captions embeddings pickle
                train_captions = self.load_captions(data_dir, train_names)
                test_captions = self.load_captions(data_dir, test_names)

                train_captions, test_captions, ixtoword, wordtoix, n_words = \
                    self.build_dictionary(train_captions, test_captions)
                with open(captions_filepath, 'wb') as f:
                    pickle.dump([train_captions, test_captions,
                                 ixtoword, wordtoix], f, protocol=2)
                    print('Save to: ', captions_filepath)

            # Pretrain embeddings
            self.pretrain_embeddings()

        # Load the pretrained embeddings
        with open(pretrained_filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions = x[0], x[1]
            train_strings, test_strings = x[2], x[3]
            del x
            print(f'Load embeddings from: {pretrained_filepath}')

        # Set the embedding dimension to the loaded file
        cfg.TEXT.EMBEDDING_DIM = len(train_captions[0])
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
            raw_strings = train_strings
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
            raw_strings = test_strings

        return filenames, captions, raw_strings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def pretrain_embeddings(self):
        captions_filepath = os.path.join(cfg.DATA_DIR, 'captions.pickle')
        out_filepath = os.path.join(cfg.DATA_DIR, f'captions_{self.encoder_type}.pickle')
        with open(captions_filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions = x[0], x[1]
            ixtoword, wordtoix = x[2], x[3]
            del x
            n_words = len(ixtoword)
            print('Load from: ', captions_filepath)

        if self.encoder_type == "SBERT":
            text_encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        elif self.encoder_type == "RNN":
            text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            text_encoder.cuda()
        else:
            print(f"Encoder type {self.encoder_type} not supported.")
            raise Exception

        # 0: Train embeds 1: Test embeds 2: Train strings 3: Test strings
        x = [[], [], [], []]

        for i, sent_ix in enumerate(tqdm(train_captions, desc="Generating train embeddings")):
            sent_indexed, sent_raw, sent_len = self.preprocess_caption(sent_ix, ixtoword)
            x[2].append(sent_raw)
            if self.encoder_type == "SBERT":
                x[0].append(text_encoder.encode(sent_raw))
            elif self.encoder_type == "RNN":
                hidden = text_encoder.init_hidden(1)
                sent_indexed = torch.from_numpy(np.reshape(sent_indexed, (1, cfg.TEXT.WORDS_NUM))).cuda()
                sent_len = torch.from_numpy(np.expand_dims(np.asarray(sent_len), axis=0)).cuda()
                with torch.no_grad():
                    words_embs, sent_emb = text_encoder(sent_indexed, sent_len, hidden)
                x[0].append(np.squeeze(sent_emb.cpu().numpy()))

        for i, sent_ix in enumerate(tqdm(test_captions,  desc="Generating test embeddings")):
            sent_indexed, sent_raw, sent_len = self.preprocess_caption(sent_ix, ixtoword)
            x[3].append(sent_raw)
            if self.encoder_type == "SBERT":
                x[1].append(text_encoder.encode(sent_raw))
            elif self.encoder_type == "RNN":
                hidden = text_encoder.init_hidden(1)
                sent_indexed = torch.from_numpy(np.reshape(sent_indexed, (1, cfg.TEXT.WORDS_NUM))).cuda()
                sent_len = torch.from_numpy(np.expand_dims(np.asarray(sent_len), axis=0)).cuda()
                with torch.no_grad():
                    words_embs, sent_emb = text_encoder(sent_indexed, sent_len, hidden)
                x[1].append(np.squeeze(sent_emb.cpu().numpy()))

        print(f"Saving embeddings to {out_filepath}")
        with open(out_filepath, "wb") as out_file:
            pickle.dump(x, out_file)

    def preprocess_caption(self, caption, ixtoword):
        # a list of indices for a sentence
        sent_caption = np.asarray(caption).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')

        sent_ix = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        sent_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            sent_ix[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            sent_ix[:, 0] = sent_caption[ix]
            sent_len = cfg.TEXT.WORDS_NUM

        sent_str = []
        for ix in sent_caption:
            sent_str.append(ixtoword[ix])
        sent_str = " ".join(sent_str)
        return sent_ix, sent_str.capitalize(), sent_len

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = np.random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        return imgs, self.captions[new_sent_ix], self.raw_strings[new_sent_ix], cls_id, key

    def __len__(self):
        return len(self.filenames)
