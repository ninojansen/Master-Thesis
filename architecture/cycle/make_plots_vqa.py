
import os
from numpy.lib.npyio import load
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import argparse
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import subprocess
import matplotlib.ticker as ticker
from tqdm import tqdm
import gc
import math


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--outdir', dest='output_dir', type=str,
                        default='/home/nino/Dropbox/Documents/Master/Thesis/Results/cycle_vqa/plots')
    args = parser.parse_args()
    return args


def load_df(experiment_id):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    runs = df["run"].unique()
  #  print(df["tag"].unique())

    df = df[df["tag"].isin(metrics) == True]
    df = df.pivot_table(values=("value"), columns="tag", index=["run", "step"])
    # `reset_index()` removes the MultiIndex structure of the pivoted
    # DataFrame. Before the call, the DataFrame consits of two levels
    # of index: "run" and "step". After the call, the index become a
    # single range index (e.g,. `dataframe[:2]` works).
    df = df.reset_index()
    df.columns.name = None
    # Remove the columns name "tag".
    df.columns.names = [None for name in df.columns.names]
    # Change step to epoch
    df["step"] = df['step'].apply(lambda x: math.floor(x / df['step'][0]))
    df["run"] = df['run'].apply(lambda x: rename(x))
    # df = df.drop(df[(df.run == "ef=sbert_full_nhidden=256_lr=0.002") |
    #                 (df.run == "ef=phoc_full_nhidden=256_lr=0.002")].index)

    # if key == "Vision only":
    #     df.rename(columns={"run": "network"}, inplace=True)
    #     hue = "network"
    # else:
    df.rename(columns={"run": "method"}, inplace=True)
   # df["text embedding"] = df["text embedding"].apply(lambda x: x[3:x.index("nhidden") - 1])
    #hue = "method"
    return df


def rename(x):
    if "cns_only" in x:
        return "Consistency only"
    elif "full_False" in x:
        return "Full"
    elif "full_True" in x:
        return "Full + Gating"
    elif "full_coeff_False":
        return "Full Coeff"
    elif "full_coeff_True":
        return "Full Coeff + Gating"
    elif "vqa_only":
        return "VQA only"


if __name__ == "__main__":
    args = parse_args()

    experiment_ids = {"Cycle 1": [
        "05qd6zf8RUWbBznr1mtLGw", "Ldn3Yxg2SOCZgsyNcXZdNQ", "XbdU1AX2RVixIws5ac5XWw"], "Cycle 2": [
        "JYWjVEEBTE6zxdAiK7b3ow", "KfDYvD9dR0SlH8YR9KlwQw", "0Ea4iJoJQMSAVEEk27j7VQ"]}

    os.makedirs(args.output_dir, exist_ok=True)

    # TODO
    # Start from zero
    # Make plots per cnn_type
    # Chnage y to epoch
    sns.set_theme()
    sns.set_context("paper")
    err_style = "band"
    hue = "method"

    metrics = ["Acc/Train_Consistency_epoch", "Acc/Train_VQA_epoch", "Acc/Val",
               "Loss/Answer_Consistency_epoch", "Loss/Image_Consistency_epoch", "Loss/Total_epoch", "Loss/VQA_epoch"]
# ['Acc/Train_Consistency_epoch'
#  'Acc/Train_VQA_epoch' 'Acc/Train_VQA_step' 'Acc/Val'
#  'Loss/Answer_Consistency_epoch'
#  'Loss/Total_epoch',
#  'Loss/Image_Consistency_epoch',
#  'Loss/VQA_epoch']

    cycle1_df = []
    cycle2_df = []
    for cycle1, cycle2 in tqdm(
            zip(experiment_ids["Cycle 1"],
                experiment_ids["Cycle 2"]),
            total=len(experiment_ids["Cycle 1"])):
        cycle1_df.append(load_df(cycle1))
        cycle2_df.append(load_df(cycle2))

    cycle1_df = pd.concat(cycle1_df)
    cycle2_df = pd.concat(cycle2_df)

    # Train consistency accuracy
    plt.figure(figsize=(8, 4))
    plt.suptitle("Train answer consistency accuracy")
    ax = plt.subplot(1, 2, 1)
    plt.ylim(0, 1)
    sns.lineplot(data=cycle1_df, x="step", y="Acc/Train_Consistency_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    ax = plt.subplot(1, 2, 2)
    plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df, x="step", y="Acc/Train_Consistency_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.savefig(f"{args.output_dir}/train_cns_acc.png", bbox_inches="tight")
    plt.clf()

    # Train VQA Accuracy

    plt.figure(figsize=(8, 4))
    plt.suptitle("Train VQA accuracy")
    ax = plt.subplot(1, 2, 1)
    plt.ylim(0, 1)
    sns.lineplot(data=cycle1_df, x="step", y="Acc/Train_VQA_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    ax = plt.subplot(1, 2, 2)
    plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df, x="step", y="Acc/Train_VQA_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.savefig(f"{args.output_dir}/train_vqa_acc.png", bbox_inches="tight")
    plt.clf()

    # Validation accuracy
    plt.figure(figsize=(8, 4))
    plt.suptitle("Validation accuracy")
    ax = plt.subplot(1, 2, 1)
    plt.ylim(0, 1)
    sns.lineplot(data=cycle1_df, x="step", y="Acc/Val", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    ax = plt.subplot(1, 2, 2)
    plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df, x="step", y="Acc/Val", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.savefig(f"{args.output_dir}/val_acc.png", bbox_inches="tight")
    plt.clf()

    # Loss answer consistency
    plt.figure(figsize=(8, 4))
    plt.suptitle("Loss answer consistency")
    ax = plt.subplot(1, 2, 1)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle1_df, x="step", y="Loss/Answer_Consistency_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    ax = plt.subplot(1, 2, 2)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df, x="step", y="Loss/Answer_Consistency_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(f"{args.output_dir}/loss_answer_consistency.png", bbox_inches="tight")
    plt.clf()

    # Loss image consistency
    plt.figure(figsize=(8, 4))
    plt.suptitle("Loss image consistency")
    ax = plt.subplot(1, 2, 1)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle1_df, x="step", y="Loss/Image_Consistency_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    ax = plt.subplot(1, 2, 2)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df, x="step", y="Loss/Image_Consistency_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(f"{args.output_dir}/loss_image_consistency.png", bbox_inches="tight")
    plt.clf()

    # Loss total
    plt.figure(figsize=(8, 4))
    plt.suptitle("Loss total")
    ax = plt.subplot(1, 2, 1)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle1_df, x="step", y="Loss/Total_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    ax = plt.subplot(1, 2, 2)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df, x="step", y="Loss/Total_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(f"{args.output_dir}/loss_total.png", bbox_inches="tight")
    plt.clf()

    # Loss vqa
    plt.figure(figsize=(8, 4))
    plt.suptitle("Loss VQA")
    ax = plt.subplot(1, 2, 1)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle1_df, x="step", y="Loss/VQA_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    ax = plt.subplot(1, 2, 2)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df, x="step", y="Loss/VQA_epoch", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(f"{args.output_dir}/loss_vqa.png", bbox_inches="tight")
    plt.clf()
