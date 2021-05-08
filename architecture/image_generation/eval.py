
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
    parser.add_argument('--ckpt', dest='ckpt', type=str,
                        default="/home/nino/Documents/Models/IG/ig_experiment_final")
    parser.add_argument('--vqa_ckpt', dest='vqa_ckpt', type=str,
                        default="/home/nino/Documents/Models/VQA/vqa_experiment_final/top_attention")
    parser.add_argument('--data_dir', dest='data_dir', type=str, default="/home/nino/Documents/Datasets/ExtEasyVQA")
    parser.add_argument('--config_name', dest='name', type=str, default="ig_results")
    parser.add_argument('--outdir', dest='output_dir', type=str,
                        default='/home/nino/Dropbox/Documents/Master/Thesis/Results/IG')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=-1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    batch_size = 8
    datamodule = EasyVQADataModule(data_dir=args.data_dir, batch_size=batch_size,
                                   num_workers=12, im_size=128, pretrained_text=True, iterator="image")
    datamodule.setup("test")

    vqa_ckpts = {}
    for name in os.listdir(args.vqa_ckpt):
        path = os.path.join(args.vqa_ckpt, name, "checkpoints")
        ckpt = os.listdir(path)[np.argmax([x[x.index("step=") + 5:x.index(".ckpt")] for x in os.listdir(path)])]
        ef_type = name[3:name.index("nhidden") - 1]
        vqa_ckpts[ef_type] = os.path.join(path, ckpt)

    model_ckpts = []
    for ef_type in os.listdir(args.ckpt):
        experiment_path = os.path.join(args.ckpt, ef_type)
        for name in os.listdir(experiment_path):
            if "vae" not in name:
                path = os.path.join(experiment_path, name, "checkpoints")
                ckpt = os.listdir(path)[np.argmax([x[x.index("step=") + 5:x.index(".ckpt")] for x in os.listdir(path)])]

                model_ckpts.append((os.path.join(path, ckpt), ef_type, name))

    df = pd.DataFrame(columns=["FID", "Inception mean", "Inception std", "VQA", "Path"],
                      index=[f"{x[1]}+{x[2]}" for x in model_ckpts])

    test_questions = []
    for batch in datamodule.test_dataloader():
        for q, a, type, specificity, text in zip(
                batch["question_json"]["question"],
                batch["question_json"]["answer"],
                batch["question_json"]["type"], batch["question_json"]["specificity"], batch["text"]):
            test_questions.append((q, a, type, specificity, text))
    # TODO make sampling based on type and spec to balance
    n_questions = 10

    # sample_questions = random.sample([x for x in test_questions if x[2] == "count"],
    #                                  2) + random.sample([x for x in test_questions if x[2] == "size"],
    #                                                     2) + random.sample([x for x in test_questions if x[2] == "shape"],
    #                                                                        2) + random.sample([x for x in test_questions if x[2] == "color"],
    #                                                                                           2) + random.sample([x for x in test_questions if x[2] == "location"],

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
  #  sample_questions = random.sample(test_questions, n_questions)

    answer_map = datamodule.get_answer_map()
    sample_images = []
    for i, (ckpt, ef_type, name) in enumerate(model_ckpts):
        print(name)
        model = DFGAN.load_from_checkpoint(ckpt)

        cfg = model.cfg
        model.vqa_model = VQA.load_from_checkpoint(vqa_ckpts[ef_type])
        model.batch_size = batch_size
        model.text_embedding_generator = TextEmbeddingGenerator(ef_type=cfg.MODEL.EF_TYPE, data_dir=args.data_dir)
        model.cuda()
        text_embeddings = []

        q_embedding = model.text_embedding_generator.process_batch([x[0] for x in sample_questions])
        a_embedding = model.text_embedding_generator.process_batch([x[1] for x in sample_questions])
        qa_embedding = torch.cat((q_embedding, a_embedding), dim=1).cuda()

        noise = torch.randn(len(sample_questions), model.cfg.MODEL.Z_DIM).cuda()
        fake_pred = model(noise, qa_embedding)
        sample_images.append(fake_pred)
       # sample_images[f"{ef_type}+{name}"] = fake_pred
        trainer = pl.Trainer.from_argparse_args(
            args, default_root_dir=args.output_dir)
        vqa_results = trainer.test(model, test_dataloaders=datamodule.test_dataloader())
        print(model.results, vqa_results)
        row_id = f"{ef_type}+{name}"
        df["FID"][row_id] = model.results["FID"]
        df["Inception mean"][row_id] = model.results["IS_MEAN"]
        df["Inception std"][row_id] = model.results["IS_STD"]
        df["VQA"][row_id] = vqa_results[0]["VQA_Acc"]
        df["Path"][row_id] = ckpt

    os.makedirs(args.output_dir, exist_ok=True)
    grid = torchvision.utils.save_image(
        torch.vstack(sample_images),
        f"{args.output_dir}/{args.name}_image.png", normalize=True, nrow=n_questions, padding=16, pad_value=255)

    with open(f"{args.output_dir}/{args.name}_image.txt", 'w') as writer:
        writer.write(" || ".join([f"{x[0]} {x[1]} type={x[2]} spec={x[3]}" for x in sample_questions]) + "\n")
        for ckpt, ef_type, name in model_ckpts:
            writer.write(f"{ef_type} {name}\n")

    df.to_csv(f"{args.output_dir}/{args.name}.csv")
