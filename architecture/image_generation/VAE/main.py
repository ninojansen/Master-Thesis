
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from architecture.datasets.easy_vqa import EasyVQADataModule
from architecture.image_generation.VAE.trainer import DCGAN, VAE
from architecture.image_generation.VAE.config import cfg, cfg_from_file
from architecture.datasets.easy_vqa import EasyVQADataModule
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image
import torchvision.transforms as transforms
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import pytorch_lightning as pl
import pprint
import argparse
from pl_bolts.datamodules import CIFAR10DataModule
#multiprocessing.set_start_method('spawn', True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/cifar10.yml', type=str)
    parser.add_argument('--datadir', dest='data_dir', type=str, default='')
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=None)
    parser.add_argument('--ckpt', dest='ckpt', type=str, default=None)
    parser.add_argument('--test', dest='test', action="store_true", default=False)
    parser.add_argument('--version', dest='version', type=str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=-1)
    parser.set_defaults(max_epochs=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    if args.max_epochs:
        cfg.TRAIN.MAX_EPOCH = args.max_epochs

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

    datamodule = None
    if cfg.DATASET_NAME == "easy_vqa":
        datamodule = EasyVQADataModule(data_dir=cfg.DATA_DIR, batch_size=cfg.TRAIN.BATCH_SIZE,
                                       num_workers=num_workers, im_size=cfg.IM_SIZE, val_split=False)
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

    if args.ckpt:
        vae_model = VAE.load_from_checkpoint(args.ckpt)
    else:
        vae_logger = TensorBoardLogger(args.output_dir, name=f'{cfg.CONFIG_NAME}_vae', version=args.version)
        vae_trainer = pl.Trainer.from_argparse_args(
            args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=vae_logger, default_root_dir=args.output_dir)
        vae_model = VAE(cfg)
        vae_trainer.fit(vae_model, datamodule)

    pre_dcgan_logger = TensorBoardLogger(args.output_dir, name=f'{cfg.CONFIG_NAME}_dcgan', version=args.version)
    pre_dcgan_trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=pre_dcgan_logger, default_root_dir=args.output_dir)

    pre_dcgan_model = DCGAN(cfg, vae_model.decoder)
    pre_dcgan_trainer.fit(pre_dcgan_model, datamodule)
    pretrained_result = pre_dcgan_trainer.test(pre_dcgan_model)

    dcgan_logger = TensorBoardLogger(args.output_dir, name=f'{cfg.CONFIG_NAME}_dcgan')
    dcgan_trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=dcgan_logger, default_root_dir=args.output_dir)
    dcgan_model = DCGAN(cfg)
    dcgan_trainer.fit(dcgan_model, datamodule)
    result = dcgan_trainer.test(dcgan_model)

    print("Pretrained result:")
    print(pretrained_result)

    print("Result:")
    print(result)

    # if args.ckpt:
    #     model = VAE.load_from_checkpoint(args.ckpt)
    # else:
    #     model = VAE(cfg)

    # if args.test:
    #     # Only test the network
    #     print("Test argument specififed; Running testing loop using specified model")
    #     datamodule.setup(stage="test")
    #     result = trainer.test(model, test_dataloaders=datamodule.test_dataloader())
    #     print(result)
    # else:
    #     # Train and test the network when finished training
    #     trainer.fit(model, datamodule)
    #     result = trainer.test(model)
    #     print(result)
