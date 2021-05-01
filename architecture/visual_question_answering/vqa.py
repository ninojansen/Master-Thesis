
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from architecture.datasets.easy_vqa import EasyVQADataModule
from architecture.datasets.abstractVQA import AbstractVQADataModule
from architecture.datasets.cub200 import CUB200DataModule
from architecture.visual_question_answering.config import cfg, cfg_from_file
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms as transforms
import torch
import numpy as np
import pytorch_lightning as pl
import pprint
import argparse
from pl_bolts.datamodules import CIFAR10DataModule
from architecture.visual_question_answering.trainer import VQA
from datetime import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/easyVQA/default.yml', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default=None)
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=None)
    parser.add_argument('--test', dest='test', action="store_true", default=False)
    # Model parameters
    # EF_TYPE, ATTENTION, CNN_TYPE, N_HIDDEN, LR
    parser.add_argument('--config_name', dest='config_name', type=str, default=None)
    parser.add_argument('--ef_type', dest='ef_type', type=str, default=None)
    parser.add_argument('--n_hidden', dest='n_hidden', type=int, default=None)
    parser.add_argument('--cnn_type', dest='cnn_type', type=str, default=None)
    parser.add_argument('--type', dest='type', type=str, default=None)
    parser.add_argument('--attention', dest='attention', action="store_true", default=False)
    parser.add_argument('--lr', dest='lr', type=float, default=None)

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
    if args.num_workers:
        cfg.N_WORKERS = args.num_workers
    if args.data_dir:
        cfg.DATA_DIR = args.data_dir
    if args.ef_type:
        cfg.MODEL.EF_TYPE = args.ef_type
    if args.n_hidden:
        cfg.MODEL.N_HIDDEN = args.n_hidden
    if args.cnn_type:
        cfg.MODEL.CNN_TYPE = args.cnn_type
    if args.attention:
        cfg.MODEL.ATTENTION = args.attention
    if args.type:
        cfg.MODEL.TYPE = args.type
    if args.lr:
        cfg.TRAIN.LR = args.lr
    if args.config_name:
        cfg.CONFIG_NAME = args.config_name

    print('Using config:')
    pprint.pprint(cfg)

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
            data_dir=cfg.DATA_DIR, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=12, im_size=cfg.IM_SIZE,
            cnn_type=cfg.MODEL.CNN_TYPE, pretrained_text=True, text_embed_type=cfg.MODEL.EF_TYPE)

    if cfg.DATASET_NAME == "abstract_vqa":
        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        datamodule = AbstractVQADataModule(
            data_dir=cfg.DATA_DIR, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.N_WORKERS, im_size=cfg.IM_SIZE,
            text_embed_type=cfg.MODEL.EF_TYPE)

    cfg.MODEL.EF_DIM = datamodule.get_ef_dim(combined=False)
    cfg.MODEL.N_ANSWERS = len(datamodule.get_answer_map())

    #version = datetime.now().strftime("%d-%m_%H:%M:%S")
    version = f"ef={cfg.MODEL.EF_TYPE}_nhidden={cfg.MODEL.N_HIDDEN}_lr={cfg.TRAIN.LR}"

    logger = TensorBoardLogger(args.output_dir, name=cfg.CONFIG_NAME, log_graph=True,
                               version=version)

    early_stop_callback = EarlyStopping(
        monitor='Acc/Val',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='max'
    )

    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=logger, default_root_dir=args.output_dir,
        auto_lr_find=True, callbacks=[early_stop_callback])

    model = VQA(cfg)
    print(f"==============Training {cfg.CONFIG_NAME} model==============")
  #  trainer.tune(model, datamodule)
    trainer.fit(model, datamodule)

    print(f"==============Validating final {cfg.CONFIG_NAME} model==============")
    result = trainer.test(model, test_dataloaders=datamodule.test_dataloader())
    print("Result:")
    print(result)
