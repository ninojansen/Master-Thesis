
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
from architecture.cycle.trainer import FinetuneVQA, FinetuneIG
from datetime import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd

from architecture.embeddings.text.generator import TextEmbeddingGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument(
        '--ckpt', dest='ckpt', type=str,
        default="/home/nino/Dropbox/Documents/Master/Thesis/architecture/cycle/output/finetune_ig/cycle_vqa_only_False_16-05_22:11:26/checkpoints/epoch=0-step=9.ckpt")
    parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/nino/Documents/Datasets/ExtEasyVQA")
    parser.add_argument('--config_name', dest='name', type=str, default="gating")
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output')

    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=-1)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    datamodule = EasyVQADataModule(
        data_dir=args.data_dir, batch_size=24, num_workers=12, im_size=128, pretrained_text=False)
    datamodule.setup("test")

    model_ckpts = [(args.ckpt, "gating")]
    # if os.path.isdir(args.ckpt):
    #     for name in os.listdir(args.ckpt):
    #         path = os.path.join(args.ckpt, name, "checkpoints")
    #         ckpt = os.listdir(path)[np.argmax([x[x.index("step=") + 5:x.index(".ckpt")] for x in os.listdir(path)])]
    #         model_ckpts.append((os.path.join(path, ckpt), name))

    df = pd.DataFrame(columns=["Full", "Yes/No", "Open", "Size", "Shape", "Color", "Location",
                               "Count", "Spec1", "Spec2", "Spec3", "Path"], index=[x[1] for x in model_ckpts])

    for i, (ckpt, name) in enumerate(model_ckpts):
        model = FinetuneIG.load_from_checkpoint(ckpt)
