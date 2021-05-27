
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
from collections import OrderedDict
from architecture.embeddings.text.generator import TextEmbeddingGenerator
import torchvision
from architecture.image_generation.trainer import DFGAN


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument(
        '--ckpt', dest='ckpt', type=str,
        default="/home/nino/Downloads/ig_cycle_final2/finetune_ig/cycle_full_False_26-05_14:10:32/checkpoints/epoch=99-step=37499.ckpt")
    parser.add_argument(
        '--ig_ckpt', dest='ig_ckpt', type=str,
        default="/home/nino/Documents/Models/IG_FINAL/sbert_reduced3/pretrained_21-05_20:36:25/checkpoints/epoch=399-step=149999.ckpt")

    parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/nino/Documents/Datasets/ExtEasyVQA")
    parser.add_argument('--name', dest='name', type=str, default="cycle_ig")
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output')

    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=-1)
    args = parser.parse_args()

    return args


def process_state_dict(x):
    res = OrderedDict()
    for key, value in x.items():
        if "ig_model.generator." in key:
            res[key.replace("ig_model.generator.", "")] = value
    return res


if __name__ == "__main__":
    args = parse_args()

    # datamodule = EasyVQADataModule(
    #     data_dir=args.data_dir, batch_size=24, num_workers=12, im_size=128, pretrained_text=False)
    # datamodule.setup("test")

    model = DFGAN.load_from_checkpoint(args.ig_ckpt)
    model.generator.load_state_dict(process_state_dict(torch.load(args.ckpt)["state_dict"]))
    model.cuda()
    text_embedding_generator = TextEmbeddingGenerator(ef_type=model.cfg.MODEL.EF_TYPE, data_dir=args.data_dir)

    count_questions = [('How many shapes are in the image?', 'two', 'count', 0),
                       ("How many black objects are in the image?", 'one', 'count', 1)]
    color_questions = [('Which color is the small cicrcle?', 'brown', 'color', 2),
                       ("Is there a orange circle present?", 'yes', 'color', 2)]
    shape_questions = [("Does the image contain a medium sized indigo circle?", 'yes', 'shape', 3),
                       ("Does the image contain a circle?", 'yes', 'shape', 1)]

    size = [("How large is the orange circle?", 'small', 'size', 3),
            ("Does the image contain a large rectangle?", 'yes', 'shape', 1)]

    location_questions = [("Is there a red triangle above the circle?", 'yes', 'location', 3),
                          ("In which part is the violet triangle placed?", 'bottom', 'location', 2)]

    sample_questions = count_questions + color_questions + shape_questions + size + location_questions
    images = []
    q_embedding = text_embedding_generator.process_batch([x[0] for x in sample_questions])
    a_embedding = text_embedding_generator.process_batch([x[1] for x in sample_questions])
    qa_embedding = torch.cat((q_embedding, a_embedding), dim=1).cuda()

    noise = torch.randn(len(sample_questions), model.cfg.MODEL.Z_DIM).cuda()
    with torch.no_grad():
        fake_pred = model(noise, qa_embedding)
    torchvision.utils.save_image(
        fake_pred, os.path.join(args.output_dir, f"test_image.png"),
        normalize=True, nrow=10, padding=16, pad_value=255)
