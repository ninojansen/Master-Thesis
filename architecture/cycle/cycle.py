
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from architecture.datasets.easy_vqa import EasyVQADataModule
from architecture.datasets.cub200 import CUB200DataModule
from architecture.cycle.config import cfg, cfg_from_file
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms as transforms
import torchvision
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
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/finetune_vqa.yml', type=str)
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=None)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default=None)
    parser.add_argument("--vqa_ckpt", dest='vqa_ckpt', type=str, default=None)
    parser.add_argument("--gating", dest='gating', action='store_true')
    parser.add_argument("--ig_ckpt", dest='ig_ckpt', type=str, default=None)
    parser.add_argument("--type", dest='type', type=str, default="vqa")
    parser.add_argument("--loss", dest='loss', type=str, default="full")
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
    if args.num_workers:
        cfg.N_WORKERS = args.num_workers
    if args.vqa_ckpt:
        cfg.MODEL.VQA_CHECKPOINT = args.vqa_ckpt
    if args.ig_ckpt:
        cfg.MODEL.IG_CHECKPOINT = args.ig_ckpt
    if args.loss:
        cfg.TRAIN.LOSS = args.loss

    cfg.TRAIN.GATING = args.gating

    vqa_model = VQA.load_from_checkpoint(cfg.MODEL.VQA_CHECKPOINT)
    ig_model = DFGAN.load_from_checkpoint(cfg.MODEL.IG_CHECKPOINT)

    if vqa_model.cfg.MODEL.EF_TYPE != ig_model.cfg.MODEL.EF_TYPE:
        raise NameError(
            f"VQA embedding type: {vqa_model.cfg.MODEL.EF_TYPE} does not match IG embedding type {ig_model.cfg.MODEL.EF_TYPE}")

    cfg.MODEL.EF_TYPE = vqa_model.cfg.MODEL.EF_TYPE

    if args.type == "ig":
        datamodule = EasyVQADataModule(
            data_dir=cfg.DATA_DIR, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.N_WORKERS, im_size=cfg.IM_SIZE,
            pretrained_text=True, text_embed_type=cfg.MODEL.EF_TYPE, iterator="image")
    else:
        datamodule = EasyVQADataModule(
            data_dir=cfg.DATA_DIR, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.N_WORKERS, im_size=cfg.IM_SIZE,
            pretrained_text=True, text_embed_type=cfg.MODEL.EF_TYPE, cnn_type=vqa_model.cfg.MODEL.CNN_TYPE,
            iterator="question")

    answer_map = datamodule.get_answer_map()

    print('Using config:')
    pprint.pprint(cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.type == "ig":
        cycle_model = FinetuneIG(cfg, vqa_model, ig_model, answer_map)
    else:
        cycle_model = FinetuneVQA(cfg, vqa_model, ig_model, answer_map)
    version = datetime.now().strftime("%d-%m_%H:%M:%S")
    logger = TensorBoardLogger(
        args.output_dir, name=f"finetune_{args.type}", version=f"cycle_{cfg.TRAIN.LOSS}_{cfg.TRAIN.GATING}_{version}")
    if args.type == "ig":
        torch.save(cycle_model.ig_model.generator.state_dict(),
                   f"{args.output_dir}/finetine_ig_{cfg.TRAIN.LOSS}_{version}_model")
    else:
        torch.save(cycle_model.vqa_model.model.state_dict(),
                   f"{args.output_dir}/finetine_vqa_{cfg.TRAIN.LOSS}_{cfg.TRAIN.GATING}_{version}_model")
    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=logger, default_root_dir=args.output_dir)

    # print(f"==============Training {cfg.CONFIG_NAME} model==============")
    trainer.fit(cycle_model, datamodule)

    result = trainer.test(cycle_model)

    if args.type == "vqa":
        df = pd.DataFrame(columns=["Full", "Yes/No", "Open", "Size", "Shape", "Color", "Location",
                                   "Count", "Spec1", "Spec2", "Spec3", "Path"], index=[f"finetune_vqa_{cfg.TRAIN.LOSS}"])
        i = 0
        df["Full"][i] = result[0]["Test/Acc/General"]
        df["Yes/No"][i] = result[0]["Test/Acc/Bool"]
        df["Open"][i] = result[0]["Test/Acc/Open"]
        df["Size"][i] = result[0]["Test/Acc/Size"]
        df["Shape"][i] = result[0]["Test/Acc/Shape"]
        df["Color"][i] = result[0]["Test/Acc/Color"]
        df["Location"][i] = result[0]["Test/Acc/Location"]
        df["Count"][i] = result[0]["Test/Acc/Count"]
        df["Spec1"][i] = result[0]["Test/Acc/Spec1"]
        df["Spec2"][i] = result[0]["Test/Acc/Spec2"]
        df["Spec3"][i] = result[0]["Test/Acc/Spec3"]
        df.to_csv(f"{args.output_dir}/finetune_vqa_{cfg.TRAIN.LOSS}_{version}.csv")
    else:

        df = pd.DataFrame(
            columns=["FID", "Inception mean", "Inception std", "VQA", "Path"],
            index=[f"finetune_ig_{cfg.TRAIN.LOSS}"])
        row_id = 0
        df["FID"][row_id] = cycle_model.results["FID"]
        df["Inception mean"][row_id] = cycle_model.results["IS_MEAN"]
        df["Inception std"][row_id] = cycle_model.results["IS_STD"]
        df["VQA"][row_id] = result[0]["Test/VQA_Acc"]
        df.to_csv(f"{args.output_dir}/finetune_ig_{cfg.TRAIN.LOSS}_{version}.csv")

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
        q_embedding = cycle_model.text_embedding_generator.process_batch([x[0] for x in sample_questions])
        a_embedding = cycle_model.text_embedding_generator.process_batch([x[1] for x in sample_questions])
        qa_embedding = torch.cat((q_embedding, a_embedding), dim=1).cuda()
        noise = torch.randn(len(sample_questions), cycle_model.ig_model.cfg.MODEL.Z_DIM).cuda()
        fake_pred = cycle_model.ig_model(noise, qa_embedding)

        grid = torchvision.utils.save_image(
            fake_pred, f"{args.output_dir}/finetune_ig_{cfg.TRAIN.LOSS}_{version}_image.png", normalize=True,
            nrow=len(sample_questions),
            padding=16, pad_value=255)

        with open(f"{args.output_dir}/finetune_ig_{cfg.TRAIN.LOSS}_{version}_image.txt", 'w') as writer:
            writer.write(" || ".join([f"{x[0]} {x[1]} type={x[2]} spec={x[3]}" for x in sample_questions]) + "\n")
            writer.write(f"finetune_ig_{cfg.TRAIN.LOSS}_{version}\n")
