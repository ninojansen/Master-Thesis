
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from architecture.datasets.easy_vqa import EasyVQADataModule
from architecture.datasets.abstractVQA import AbstractVQADataModule
from architecture.datasets.cub200 import CUB200DataModule
from architecture.visual_question_answering.config import cfg, cfg_from_file
from architecture.visual_question_answering.trainer import VQA
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms as transforms
import torch
import numpy as np
import pytorch_lightning as pl
import pprint
import argparse
from pl_bolts.datamodules import CIFAR10DataModule
from architecture.image_generation.trainer import DFGAN
from datetime import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import random
from architecture.embeddings.text.generator import TextEmbeddingGenerator
import matplotlib.pyplot as plt
import torchvision


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument(
        '--ckpt', dest='ckpt', type=str,
        default="/home/nino/Documents/Models/IG/ig_experiment_final/phoc_reduced/non_pretrained_05-05_13:00:37/checkpoints/epoch=399-step=149999.ckpt")
    parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/nino/Documents/Datasets/ExtEasyVQA")
    parser.add_argument('--config_name', dest='name', type=str, default="ig_results")
    parser.add_argument('--outdir', dest='output_dir', type=str,
                        default='/home/nino/Dropbox/Documents/Master/Thesis/Results')

    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=-1)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # batch_size = 4
    # datamodule = EasyVQADataModule(
    #     data_dir=args.data_dir, batch_size=batch_size, num_workers=12, im_size=128, pretrained_text=True)
    # datamodule.setup("test")

    # test_questions = []
    # for batch in datamodule.test_dataloader():
    #     for q, a, type, specificity, text in zip(
    #             batch["question_json"]["question"],
    #             batch["question_json"]["answer"],
    #             batch["question_json"]["type"], batch["question_json"]["specificity"], batch["text"]):
    #         test_questions.append((q, a, type, specificity, text))
    # # TODO make sampling based on type and spec to balance
    # n_questions = 10
    # sample_questions = random.sample(test_questions, n_questions)

    #answer_map = datamodule.get_answer_map()

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

    model = DFGAN.load_from_checkpoint(args.ckpt)
    model.cuda()

    text_embedding_generator = TextEmbeddingGenerator(ef_type=model.cfg.MODEL.EF_TYPE, data_dir=args.data_dir)

    retried_images = []
    noise_images = []
    q_embedding = text_embedding_generator.process_batch([x[0] for x in sample_questions])
    a_embedding = text_embedding_generator.process_batch([x[1] for x in sample_questions])
    qa_embedding = torch.cat((q_embedding, a_embedding), dim=1).cuda()

    for _ in range(5):
        noise = torch.randn(len(sample_questions), model.cfg.MODEL.Z_DIM).cuda()
        with torch.no_grad():
            fake_pred = model(noise, qa_embedding)
        retried_images.append(fake_pred)

    for _ in range(5):
        noise = torch.randn(len(sample_questions), model.cfg.MODEL.Z_DIM).cuda()
        with torch.no_grad():
            fake_pred = model(noise, qa_embedding + torch.rand_like(qa_embedding).type_as(qa_embedding))
        noise_images.append(fake_pred)

    retried_grid = torchvision.utils.make_grid(torch.vstack(retried_images).cpu(),
                                               normalize=True,
                                               nrow=len(sample_questions),
                                               padding=16, pad_value=255)

    noise_grid = torchvision.utils.make_grid(torch.vstack(noise_images).cpu(),
                                             normalize=True,
                                             nrow=len(sample_questions),
                                             padding=16, pad_value=255)

    out = f"{args.output_dir}/{args.name}_retried_image.png"

    plt.imshow(noise_grid.permute(1, 2, 0))
    plt.show()
    print()
    # TODO Add headers and rows manually

    # os.makedirs(args.output_dir, exist_ok=True)
    # df.to_csv(f"{args.output_dir}/{args.name}.csv")
