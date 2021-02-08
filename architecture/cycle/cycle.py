
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from architecture.datasets.easy_vqa import EasyVQADataModule
from architecture.datasets.cub200 import CUB200DataModule
from architecture.cycle.config import cfg, cfg_from_file
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms as transforms
import torch
import numpy as np
import pytorch_lightning as pl
import pprint
import argparse
from pl_bolts.datamodules import CIFAR10DataModule
from architecture.visual_question_answering.trainer import VQA
from architecture.image_generation.VAE.trainers.dfgan_trainer import DFGAN
from datetime import datetime
from architecture.cycle.trainer import Cycle


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/easy_vqa.yml', type=str)
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=None)
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
    answer_map = None
    if cfg.DATASET_NAME == "easy_vqa":
        datamodule = EasyVQADataModule(
            data_dir=cfg.DATA_DIR, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=num_workers,
            pretrained_images=True, pretrained_text=True, text_embed_type=cfg.MODEL.EF_TYPE)

        answer_map = datamodule.get_answer_map()
    version = datetime.now().strftime("%d-%m_%H:%M:%S")

    vqa_model = VQA.load_from_checkpoint(cfg.VQA.CHECKPOINT)
    ig_model = DFGAN.load_from_checkpoint(cfg.IG.CHECKPOINT)

    cycle_model = Cycle(cfg, vqa_model, ig_model, answer_map)

    logger = TensorBoardLogger(args.output_dir, name=cfg.CONFIG_NAME, version=f"cycle_{version}")
    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=logger, default_root_dir=args.output_dir,
        automatic_optimization=False)

    # trainer.tune(model)
    # print(f"==============Training {cfg.CONFIG_NAME} model==============")
    trainer.fit(cycle_model, datamodule)
    # result = trainer.test(model)

    # print("Result:")
    # print(result)