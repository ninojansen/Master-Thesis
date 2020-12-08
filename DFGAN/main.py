import argparse
import multiprocessing
import os
import pprint
import pytorch_lightning as pl
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import CUB200DataModule
from misc.config import cfg, cfg_from_file
from misc.utils import mkdir_p
from trainer import DFGAN
from pytorch_lightning.callbacks import GPUStatsMonitor
#multiprocessing.set_start_method('spawn', True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--datadir', dest='data_dir', type=str, default='')
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=None)
    parser.add_argument('--ckpt', dest='ckpt', type=str, default=None)
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

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    if args.max_epochs:
        cfg.TRAIN.MAX_EPOCH = args.max_epochs
    print('Using config:')
    pprint.pprint(cfg)

    pl.utilities.seed.seed_everything(seed=1996)
    if args.num_workers:
        num_workers = args.num_workers
    elif args.gpus == -1:
        num_workers = 4 * torch.cuda.device_count()
    else:
        num_workers = 4 * args.gpus

  #  num_workers = multiprocessing.cpu_count()
    CUB200 = CUB200DataModule(data_dir=cfg.DATA_DIR, num_workers=num_workers)

    logger = TensorBoardLogger(args.output_dir, name=cfg.CONFIG_NAME)
    # --fast_dev_run // Does 1 batch and 1 epoch for quick
    # --precision 16 // for 16-bit precision
    # --progress_bar_refresh_rate 0  // Disable progress bar
    # --num_nodes 2 // Num of pereregrine nodes
    # --gpus 1 // Num of gpus per node
    # --accelerator ddp // Accelerator to train across multiple nodes on peregrine

    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=logger, log_gpu_memory=True,
        automatic_optimization=False, default_root_dir=args.output_dir)

    if args.ckpt:
        model = DFGAN.load_from_checkpoint(args.ckpt)
    else:
        model = DFGAN(cfg)

    if args.test:
        # Only test the network
        print("Test argument specififed; Running testing loop using specified model")
        CUB200.setup(stage="test")
        result = trainer.test(model, test_dataloaders=CUB200.test_dataloader())
        print(result)
    else:
        # Train and test the network when finished training
        trainer.fit(model, CUB200)
        result = trainer.test(model)
        print(result)
