
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
from architecture.image_generation.trainer import DFGAN
from datetime import datetime
from architecture.cycle.trainer import FinetuneVQA, FinetuneIG


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/easy_vqa.yml', type=str)
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=None)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default=None)
    parser.add_argument('--type', dest='type', type=str, default=None)
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

    if args.data_dir:
        cfg.DATA_DIR = args.data_dir
    if args.type:
        cfg.MODEL.TYPE = args.type

    if args.num_workers:
        cfg.N_WORKERS = args.num_workers

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

    vqa_model = VQA.load_from_checkpoint(cfg.MODEL.VQA)
    ig_model = DFGAN.load_from_checkpoint(cfg.MODEL.IG)

    if vqa_model.cfg.MODEL.EF_TYPE != ig_model.cfg.MODEL.EF_TYPE:
        raise NameError(
            f"VQA embedding type: {vqa_model.cfg.MODEL.EF_TYPE} does not match IG embedding type {ig_model.cfg.MODEL.EF_TYPE}")

    cfg.MODEL.EF_TYPE = vqa_model.cfg.MODEL.EF_TYPE
    if cfg.DATASET_NAME == "easy_vqa":
        if cfg.TRAIN.TYPE == "finetune_vqa":
            iterator = "question"
        else:
            iterator = "image"
        datamodule = EasyVQADataModule(
            data_dir=cfg.DATA_DIR, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.N_WORKERS, im_size=cfg.IM_SIZE,
            pretrained_text=True, text_embed_type=cfg.MODEL.EF_TYPE, cnn_type=vqa_model.cfg.MODEL.CNN_TYPE, iterator=iterator)

        answer_map = datamodule.get_answer_map()

    print('Using config:')
    pprint.pprint(cfg)
    version = datetime.now().strftime("%d-%m_%H:%M:%S")
    if cfg.TRAIN.TYPE == "finetune_vqa":
        cycle_model = FinetuneVQA(cfg, vqa_model, ig_model, answer_map)
    else:
        cycle_model = FinetuneIG(cfg, vqa_model, ig_model, answer_map)
    logger = TensorBoardLogger(args.output_dir, name=cfg.CONFIG_NAME, version=f"cycle_{version}")
    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=logger, default_root_dir=args.output_dir)

    # trainer.tune(model)
    # print(f"==============Training {cfg.CONFIG_NAME} model==============")
    trainer.fit(cycle_model, datamodule)
    # result = trainer.test(model)

    # print("Result:")
    # print(result)
