
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
from typing import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/finetune_vqa.yml', type=str)
    parser.add_argument('--outdir', dest='output_dir', type=str, default='./output')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=None)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default=None)
    parser.add_argument("--vqa_ckpt", dest='vqa_ckpt', type=str, default=None)
    parser.add_argument("--finetune_ckpt", dest='finetune_ckpt', type=str, default=None)
    parser.add_argument("--gating", dest='gating', action='store_true')
    parser.add_argument("--ig_ckpt", dest='ig_ckpt', type=str, default=None)
    parser.add_argument("--type", dest='type', type=str, default="vqa")
    parser.add_argument("--loss", dest='loss', type=str, default="full")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=-1)
    parser.set_defaults(max_epochs=None)

    args = parser.parse_args()
    return args


def load_vqa_results_file(path):
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["Loss", "Gating", "Full", "Yes/No", "Open", "Size", "Shape",
                                   "Color", "Location", "Count", "Spec0", "Spec1", "Spec2", "Spec3", "Path"])
    return df


def load_ig_results_file(path):
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["Loss", "Gating", "FID", "Inception", "VQA", "Path"])
    return df


def load_finetune_ckpt(path):
    x = torch.load(path)["state_dict"]
    ig_keys = [x for x in x.keys() if "ig_model" in x]
    [x.pop(key) for key in ig_keys]
    res = OrderedDict()
    for key, value in x.items():
        res[key.replace("vqa_model.model.", "")] = value
    return res


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

    if args.finetune_ckpt:
        vqa_model.model.load_from_checkpoint(load_finetune_ckpt(args.finetune_ckpt))

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
        model_save_path = f"{args.output_dir}/finetine_ig_{cfg.TRAIN.LOSS}_{version}_model"
        torch.save(cycle_model.ig_model.generator.state_dict(), model_save_path)
    else:
        model_save_path = f"{args.output_dir}/finetine_vqa_{cfg.TRAIN.LOSS}_{cfg.TRAIN.GATING}_{version}_model"
        torch.save(cycle_model.vqa_model.model.state_dict(),
                   model_save_path)
    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=cfg.TRAIN.MAX_EPOCH, logger=logger, default_root_dir=args.output_dir)

    # print(f"==============Training {cfg.CONFIG_NAME} model==============")
    trainer.fit(cycle_model, datamodule)

    result = trainer.test(cycle_model)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.type == "vqa":
        results_path = os.path.join(args.output_dir, f"finetune_vqa_results.csv")
        df = load_vqa_results_file(results_path)
        # df = pd.DataFrame(columns=["Full", "Yes/No", "Open", "Size", "Shape", "Color", "Location",
        #                            "Count", "Spec1", "Spec2", "Spec3", "Path"], index=[f"finetune_vqa_{cfg.TRAIN.LOSS}"])
        # i = 0
        row = {
            "Loss": cfg.TRAIN.LOSS,
            "Gating": cfg.TRAIN.GATING,
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
            "Spec3": result[0]["Test/Acc/Spec3"],
            "Path": model_save_path}
        df.loc[datetime.now().strftime("%d-%m_%H:%M:%S")] = row
        df.to_csv(results_path, index=False)
    else:
        ft_str = "ftvqa" if args.finetune_ckpt else ""
        results_path = os.path.join(args.output_dir, f"finetune_ig_{ft_str}_results.csv")
        df = load_ig_results_file(results_path)
        row = {
            "Loss": cfg.TRAIN.LOSS,
            "Gating": cfg.TRAIN.GATING,
            "FID": cycle_model.results["FID"],
            "Inception": cycle_model.results["IS_MEAN"],
            "VQA": result[0]["Test/VQA_Acc"],
            "Path": model_save_path}

        df.loc[datetime.now().strftime("%d-%m_%H:%M:%S")] = row
        df.to_csv(results_path, index=False)
        # count_questions = [('How many shapes are in the image?', 'two', 'count', 0),
        #                    ("How many black objects are in the image?", 'one', 'count', 1)]
        # color_questions = [('Which color is the small cicrcle?', 'brown', 'color', 2),
        #                    ("Is there a orange circle present?", 'yes', 'color', 2)]
        # shape_questions = [("Does the image contain a medium sized indigo circle?", 'yes', 'shape', 3),
        #                    ("Does the image contain a circle?", 'yes', 'shape', 1)]

        # size = [("How large is the orange circle?", 'small', 'size', 3),
        #         ("Does the image contain a large rectangle?", 'yes', 'shape', 1)]

        # location_questions = [("Is there a red triangle above the circle?", 'yes', 'location', 3),
        #                       ("In which part is the violet triangle placed?", 'bottom', 'location', 2)]

        # sample_questions = count_questions + color_questions + shape_questions + size + location_questions
        # q_embedding = cycle_model.text_embedding_generator.process_batch([x[0] for x in sample_questions])
        # a_embedding = cycle_model.text_embedding_generator.process_batch([x[1] for x in sample_questions])
        # qa_embedding = torch.cat((q_embedding, a_embedding), dim=1).cuda()
        # noise = torch.randn(len(sample_questions), cycle_model.ig_model.cfg.MODEL.Z_DIM).cuda()
        # fake_pred = cycle_model.ig_model(noise, qa_embedding)

        # grid = torchvision.utils.save_image(
        #     fake_pred, f"{args.output_dir}/finetune_ig_{cfg.TRAIN.LOSS}_{version}_image.png", normalize=True,
        #     nrow=len(sample_questions),
        #     padding=16, pad_value=255)

        # with open(f"{args.output_dir}/finetune_ig_{cfg.TRAIN.LOSS}_{version}_image.txt", 'w') as writer:
        #     writer.write(" || ".join([f"{x[0]} {x[1]} type={x[2]} spec={x[3]}" for x in sample_questions]) + "\n")
        #     writer.write(f"finetune_ig_{cfg.TRAIN.LOSS}_{version}\n")
