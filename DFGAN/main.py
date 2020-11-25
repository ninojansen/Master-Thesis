from misc.utils import mkdir_p
from misc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import train, sampling
import os
import pprint
import argparse
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG, NetD
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('-hp', dest='hide_progress', action='store_true', default=False)
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='./')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    print('Using config:')
    pprint.pprint(cfg)

    ##########################################################################
    num_worker = 4 * torch.cuda.device_count()
    torch.cuda.set_device(0)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform, encoder_type=cfg.TEXT.ENCODER)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=num_worker, pin_memory=True)
    else:
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform, encoder_type=cfg.TEXT.ENCODER)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=num_worker, pin_memory=True)

    # validation data #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = NetG(cfg.TRAIN.NF, 100, cfg.TEXT.EMBEDDING_DIM, cfg.TREE.BASE_SIZE).to(device)
    netD = NetD(cfg.TRAIN.NF, cfg.TEXT.EMBEDDING_DIM,  cfg.TREE.BASE_SIZE).to(device)

    state_epoch = 0

    if cfg.B_VALIDATION:
        count = sampling(netG, dataloader, device)  # generate images for the whole valid dataset
    else:
        train(args.output_dir, dataloader, netG, netD, state_epoch,
              batch_size, device, hide_progress=args.hide_progress)
