import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from misc.utils import mkdir_p
from misc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from DAMSM import RNN_ENCODER
from trainer import train
import os
import sys
import time
import random
import pprint
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG, NetD
import torchvision.utils as vutils
import multiprocessing
from architecture.datasets.cub200 import CUB200DataModule


multiprocessing.set_start_method('spawn', True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--pretrain', dest='pretrain', action="store_true", default=False)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def sampling(text_encoder, netG, dataloader, device):
    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    # Build and load the generator
    netG.load_state_dict(torch.load('models/%s/netG.pth' % (cfg.CONFIG_NAME)))
    netG.eval()

    batch_size = cfg.TRAIN.BATCH_SIZE
    s_tmp = model_dir
    save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(save_dir)
    cnt = 0
    for i in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for step, data in enumerate(dataloader, 0):
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            cnt += batch_size
            if step % 100 == 0:
                print('step: ', step)
            # if step > 50:
            #     break
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            #######################################################
            # (2) Generate fake images
            ######################################################
            with torch.no_grad():
                noise = torch.randn(batch_size, 100)
                noise = noise.to(device)
                fake_imgs = netG(noise, sent_emb)
            for j in range(batch_size):
                s_tmp = '%s/single/%s' % (save_dir, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_%3d.png' % (s_tmp, i)
                im.save(fullpath)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    print('Using config:')
    pprint.pprint(cfg)

    ##########################################################################

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    if args.pretrain:
        datamodule = CUB200DataModule(
            data_dir=cfg.DATA_DIR, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=int(cfg.WORKERS),
            embedding_type="RNN", im_size=cfg.TREE.BASE_SIZE)
        datamodule.setup(stage="fit")
        dataloader = datamodule.train_dataloader()
        text_encoder = None
    else:
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
        text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        text_encoder.cuda()
        for p in text_encoder.parameters():
            p.requires_grad = False
        text_encoder.eval()
    # validation data #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = NetG(cfg.TRAIN.NF, 100, cfg.TEXT.EMBEDDING_DIM).to(device)
    netD = NetD(cfg.TRAIN.NF, cfg.TEXT.EMBEDDING_DIM).to(device)

    state_epoch = 0

    version = datetime.now().strftime("%d-%m_%H:%M:%S")
    if not os.path.isdir("./output"):
        os.mkdir("./output")
    output_dir = f"./output/{cfg.CONFIG_NAME}_{version}"
    if cfg.B_VALIDATION:
        count = sampling(text_encoder, netG, dataloader, device)  # generate images for the whole valid dataset
        print('state_epoch:  %d' % (state_epoch))
    else:
        train(output_dir, dataloader, netG, netD, text_encoder, state_epoch, batch_size, device, args.pretrain)
