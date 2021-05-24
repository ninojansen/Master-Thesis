
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
import pandas as pd

from architecture.embeddings.text.generator import TextEmbeddingGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--ckpt', dest='ckpt', type=str,
                        default="/home/nino/Documents/Models/VQA/vqa_experiment_final/cnn")
    parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/nino/Documents/Datasets/ExtEasyVQA")
    parser.add_argument('--config_name', dest='name', type=str, default="")
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output')

    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=-1)
    args = parser.parse_args()

    return args


def load_results_file(path):
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["Name", "CNN Type", "Text Embedding", "Full", "Yes/No", "Open",
                                   "Size", "Shape", "Color", "Location", "Count", "Spec1", "Spec2", "Spec3"])
    return df


if __name__ == "__main__":
    args = parse_args()

    datamodule = EasyVQADataModule(
        data_dir=args.data_dir, batch_size=24, num_workers=12, im_size=128, pretrained_text=False)
    datamodule.setup("test")

    model_ckpts = []
    if os.path.isdir(args.ckpt):
        for name in os.listdir(args.ckpt):
            path = os.path.join(args.ckpt, name, "checkpoints")
            ckpt = os.listdir(path)[np.argmax([x[x.index("step=") + 5:x.index(".ckpt")] for x in os.listdir(path)])]
            model_ckpts.append((os.path.join(path, ckpt), name))

    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, f"vqa_results.csv")
    df = load_results_file(results_path)
    for i, (ckpt, name) in enumerate(model_ckpts):
        model = VQA.load_from_checkpoint(ckpt)
        cfg = model.cfg
        model.text_embedding_generator = TextEmbeddingGenerator(ef_type=cfg.MODEL.EF_TYPE, data_dir=args.data_dir)

        trainer = pl.Trainer.from_argparse_args(
            args, default_root_dir=args.output_dir)
        result = trainer.test(model, test_dataloaders=datamodule.test_dataloader())

        row = {
            "Name": name,
            "CNN Type": cfg.MODEL.CNN_TYPE,
            "Text Embedding": cfg.MODEL.EF_TYPE,
            "Full": result[0]["Test/Acc/General"],
            "Yes/No": result[0]["Test/Acc/Bool"],
            "Open": result[0]["Test/Acc/Open"],
            "Size": result[0]["Test/Acc/Size"],
            "Shape": result[0]["Test/Acc/Shape"],
            "Color": result[0]["Test/Acc/Color"],
            "Location": result[0]["Test/Acc/Location"],
            "Count": result[0]["Test/Acc/Count"],
            "Spec0": result[0]["Test/Acc/Spec0"],
            "Spec1": result[0]["Test/Acc/Spec1"],
            "Spec2": result[0]["Test/Acc/Spec2"],
            "Spec3": result[0]["Test/Acc/Spec3"]}

        df.loc[datetime.now().strftime("%d-%m_%H:%M:%S")] = row

    df.to_csv(results_path, index=False)
