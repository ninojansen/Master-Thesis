
import os
import sys
from typing import OrderedDict

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
from architecture.cycle.trainer import FinetuneVQA
from datetime import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
from architecture.visual_question_answering.trainer import VQA
from architecture.embeddings.text.generator import TextEmbeddingGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--ckpt', dest='ckpt', type=str,
                        default="/home/nino/Downloads/vqa_cycle_ftig_final1/finetune_vqa")
    parser.add_argument(
        '--vqa_ckpt', dest='vqa_ckpt', type=str,
        default="/home/nino/Downloads/vqa_cycle_final1/ef=sbert_reduced_nhidden=256_lr=0.002/checkpoints/epoch=8-step=33749.ckpt")
    parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/nino/Documents/Datasets/ExtEasyVQA")
    parser.add_argument('--name', dest='name', type=str, default="cycle1")
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output/vqa_cycle_ftig/')

    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=-1)
    args = parser.parse_args()

    return args


def load_vqa_results_file(path):
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["Loss", "Gating", "Full", "Yes/No", "Open", "Size", "Shape",
                                   "Color", "Location", "Count", "Spec0", "Spec1", "Spec2", "Spec3", "Path"])
    return df


def process_state_dict(x):
    ig_keys = [x for x in x.keys() if "ig_model" in x]
    [x.pop(key) for key in ig_keys]
    res = OrderedDict()
    for key, value in x.items():
        res[key.replace("vqa_model.model.", "")] = value
    return res


if __name__ == "__main__":
    args = parse_args()

    datamodule = EasyVQADataModule(
        data_dir=args.data_dir, batch_size=128, num_workers=12, im_size=128, pretrained_text=False)
    datamodule.setup("test")

    model = VQA.load_from_checkpoint(args.vqa_ckpt)
    cfg = model.cfg
    model.text_embedding_generator = TextEmbeddingGenerator(ef_type=cfg.MODEL.EF_TYPE, data_dir=args.data_dir)

    trainer = pl.Trainer.from_argparse_args(
        args, default_root_dir=args.output_dir)

    model_ckpts = []
    if os.path.isdir(args.ckpt):
        for name in os.listdir(args.ckpt):
            path = os.path.join(args.ckpt, name, "checkpoints")
            ckpt = os.listdir(path)[np.argmax([x[x.index("step=") + 5:x.index(".ckpt")] for x in os.listdir(path)])]
            model_ckpts.append((os.path.join(path, ckpt), name))

    results_path = os.path.join(args.output_dir, f"{args.name}_results.csv")
    df = load_vqa_results_file(results_path)
    os.makedirs(args.output_dir, exist_ok=True)

    for (path, name) in model_ckpts:
        model.model.load_state_dict(process_state_dict(torch.load(path)["state_dict"]))

        trainer = pl.Trainer.from_argparse_args(
            args, default_root_dir=args.output_dir)
        result = trainer.test(model, test_dataloaders=datamodule.test_dataloader())

        split = name.split("_")
        gating = split[3]
        loss = f"{split[1]}_{split[2]}"
        row = {
            "Loss": loss,
            "Gating": gating,
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
            "Spec3": result[0]["Test/Acc/Spec3"], }
        df.loc[datetime.now().strftime("%d-%m_%H:%M:%S")] = row

    df.to_csv(results_path, index=False)
