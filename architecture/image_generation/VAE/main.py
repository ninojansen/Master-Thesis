
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from architecture.datasets.easy_vqa import EasyVQADataModule
from architecture.datasets.cub200 import CUB200DataModule
from architecture.image_generation.VAE.config import cfg, cfg_from_file
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms as transforms
import torch
import numpy as np
import pytorch_lightning as pl
import pprint
import argparse
from pl_bolts.datamodules import CIFAR10DataModule
from architecture.image_generation.VAE.trainers.dcgan_trainer import *
from architecture.image_generation.VAE.trainers.dfgan_trainer import *
from architecture.image_generation.VAE.trainers.wgan_trainer import *
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/easy_vqa_sbert.yml', type=str)
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=None)
    parser.add_argument('--ckpt', dest='ckpt', type=str, default=None)
    parser.add_argument('--gan', dest='gan', type=str, default=None)
    parser.add_argument('--type', dest='type', type=str, default="all")
    parser.add_argument('--test', dest='test', action="store_true", default=False)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=-1)
    parser.set_defaults(max_epochs=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.max_epochs:
        cfg.TRAIN.MAX_EPOCH = args.max_epochs

    if args.gan:
        cfg.MODEL.GAN = args.gan
    print('Using config:')
    pprint.pprint(cfg)

    if args.num_workers:
        num_workers = args.num_workers
    elif args.gpus == -1:
        num_workers = 4 * torch.cuda.device_count()
    else:
        num_workers = 4 * args.gpus

    # --fast_dev_run // Does 1 batch and 1 epoch for quick
    # --precision 16 // for 16-bit precision
    # --progress_bar_refresh_rate 0  // Disable progress bar
    # --num_nodes 2 // Num of pereregrine nodes
    # --gpus 1 // Num of gpus per node
    # --accelerator ddp // Accelerator to train across multiple nodes on peregrine
    # --limit_train_batches 0.1 // Limits the amount of batches for quick training
    # --check_val_every_n_epoch 5 // Set the validation interval

    # Load the datamodule
    datamodule = None
    if cfg.DATASET_NAME == "easy_vqa":
        datamodule = EasyVQADataModule(
            data_dir=cfg.DATA_DIR, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=num_workers, im_size=cfg.IM_SIZE,
            pretrained_text=True, text_embed_type=cfg.MODEL.EF_TYPE, iterator="img")
    elif cfg.DATASET_NAME == "cifar10":
        datamodule = CIFAR10DataModule(data_dir=cfg.DATA_DIR, batch_size=cfg.TRAIN.BATCH_SIZE,
                                       num_workers=num_workers)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(int(cfg.IM_SIZE * 76 / 64)),
            transforms.RandomCrop(cfg.IM_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        datamodule.train_transforms = transform
        datamodule.test_transforms = transform
        datamodule.val_transforms = transform
    elif cfg.DATASET_NAME == "CUB200":
        datamodule = CUB200DataModule(
            data_dir=cfg.DATA_DIR, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=num_workers,
            embedding_type=cfg.MODEL.EF_TYPE, im_size=cfg.IM_SIZE)

    cfg.MODEL.EF_DIM = datamodule.get_ef_dim(combined=True)
    # Initialize loggers
    # version = 1
    # if os.path.isdir(os.path.join(args.output_dir, cfg.CONFIG_NAME)):
    #     version = len(os.listdir(os.path.join(args.output_dir, cfg.CONFIG_NAME))) + 1

    version = datetime.now().strftime("%d-%m_%H:%M:%S")
    if args.type == "all" or args.type == "vae" or args.type == "pretrain":
        # VAE training
        vae_logger = vae_logger = TensorBoardLogger(
            args.output_dir, name=cfg.CONFIG_NAME, version=f"{cfg.CONFIG_NAME}_vae_{version}")
        vae_trainer = vae_trainer = pl.Trainer.from_argparse_args(
            args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=vae_logger, default_root_dir=args.output_dir)

        if cfg.MODEL.GAN == "DFGAN":
            vae_model = VAE_DFGAN(cfg)
        elif cfg.MODEL.GAN == "WGAN":
            vae_model = VAE_WGAN(cfg)
        elif cfg.MODEL.GAN == "DCGAN":
            vae_model = VAE(cfg)
        else:
            vae_model = None
            print(f"GAN model {cfg.MODEL.GAN} not supported.")

        if cfg.TRAIN.VAE_CHECKPOINT:
            print(f"==============Using VAE model from checkpoint==============")
            vae_model.load_from_checkpoint(cfg.TRAIN.VAE_CHECKPOINT)
        else:
            print(f"==============Training VAE model for pretraining==============")
            vae_trainer.fit(vae_model, datamodule)

        if args.type == "all" or args.type == "pretrain":
            pretrained_logger = TensorBoardLogger(args.output_dir, name=cfg.CONFIG_NAME,
                                                  version=f"{cfg.CONFIG_NAME}_pretrained_{version}")
            pretrained_trainer = pl.Trainer.from_argparse_args(
                args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=pretrained_logger, automatic_optimization=False,
                default_root_dir=args.output_dir)

            if cfg.MODEL.GAN == "DFGAN":
                pretrained_model = DFGAN(cfg, vae_model.decoder)
            elif cfg.MODEL.GAN == "WGAN":
                pretrained_model = WGAN(cfg, vae_model.decoder)
            elif cfg.MODEL.GAN == "DCGAN":
                pretrained_model = DCGAN(cfg, vae_model.decoder)
            else:
                pretrained_model = None
                print(f"GAN model {cfg.MODEL.GAN} not supported.")

            print(f"==============Training {cfg.CONFIG_NAME} model with pretraining==============")
            pretrained_trainer.fit(pretrained_model, datamodule)
            pretrained_result = pretrained_trainer.test(pretrained_model)
            print("Pretrained result:")
            print(pretrained_result)

    if args.type == "all" or args.type == "no_pretrain":
        full_logger = TensorBoardLogger(args.output_dir, name=cfg.CONFIG_NAME,
                                        version=f"{cfg.CONFIG_NAME}_non_pretrained_{version}")
        full_trainer = pl.Trainer.from_argparse_args(
            args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=full_logger, automatic_optimization=False,
            default_root_dir=args.output_dir)

        if cfg.MODEL.GAN == "DFGAN":
            full_model = DFGAN(cfg)
        elif cfg.MODEL.GAN == "WGAN":
            full_model = WGAN(cfg)
        elif cfg.MODEL.GAN == "DCGAN":
            full_model = DCGAN(cfg)
        else:
            full_model = None
            print(f"GAN model {cfg.MODEL.GAN} not supported.")

        print(f"==============Training {cfg.CONFIG_NAME} model without pretraining==============")
        full_trainer.fit(full_model, datamodule)
        full_result = full_trainer.test(full_model)

        print("Non-Pretrained Result:")
        print(full_result)
