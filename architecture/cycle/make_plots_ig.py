
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
                        default='/home/nino/Dropbox/Documents/Master/Thesis/Results/cycle_ig/plots')
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

    single_step_metrics = ["Acc/Train_VQA",
                           "Loss/Image_Consistency", "Loss/Total", "Loss/VQA"]
    epoch_step_metrics = ["Acc/Val_VQA", "FID/Val", "Inception/Val_Mean", ]

    df["run"] = df['run'].apply(lambda x: rename(x))
    df.rename(columns={"run": "method"}, inplace=True)

    epoch_step = df.drop(single_step_metrics, 1).dropna()
    epoch_step["step"] = epoch_step['step'].apply(lambda x: math.floor(x / epoch_step['step'][7]))

    single_step = df.drop(epoch_step_metrics, 1)
    single_step = single_step[single_step["Loss/Total"].isna() == False]
    single_step = single_step.groupby(np.arange(len(single_step)) // 7.5).mean()

    if len(single_step) > len(epoch_step):
        single_step = single_step.drop(index=single_step.index[-(len(single_step) - len(epoch_step)):], axis=0)
    single_step["method"] = epoch_step["method"].values[:len(single_step)]
    single_step["step"] = epoch_step["step"].values

    #epoch_step["index"] = epoch_step["step"]

    # df["step"] = df['step'].apply(lambda x: math.floor(x / df['step'][0]))
    # df["run"] = df['run'].apply(lambda x: rename(x))

    return single_step, epoch_step


def rename(x):
    if "full_False" in x:
        return "Full"
    elif "vqa_only" in x:
        return "VQA only"


if __name__ == "__main__":
    args = parse_args()

    experiment_ids = {"Cycle 1": [
        "IxjQqaoFRS636nqOfo85FA", "MzMW1P4JSk6ioGuKM1A3FQ", "U8ADI1WrSrOCp50Iregyjw"], "Cycle 2": [
        "lYMzSO0HRHqoAO81SQXxRg", "MtPQw45ARmevhJbyUeu1Qw", "PnQAgxY8QGO40C865XV3Kw"]}

    os.makedirs(args.output_dir, exist_ok=True)

    sns.set_theme()
    sns.set_context("paper")
    err_style = "band"
    hue = "method"

    metrics = ["Acc/Train_VQA", "Acc/Val_VQA", "FID/Val", "Inception/Val_Mean",
               "Loss/Image_Consistency", "Loss/Total", "Loss/VQA"]

    cycle1_df_step = []
    cycle1_df_epoch = []
    cycle2_df_step = []
    cycle2_df_epoch = []
    for cycle1, cycle2 in tqdm(
            zip(experiment_ids["Cycle 1"],
                experiment_ids["Cycle 2"]),
            total=len(experiment_ids["Cycle 1"])):
        step, epoch = load_df(cycle1)
        cycle1_df_step.append(step)
        cycle1_df_epoch.append(epoch)

        step, epoch = load_df(cycle2)
        cycle2_df_step.append(step)
        cycle2_df_epoch.append(epoch)

    cycle1_df_step = pd.concat(cycle1_df_step)
    cycle1_df_epoch = pd.concat(cycle1_df_epoch)
    cycle2_df_step = pd.concat(cycle2_df_step)
    cycle2_df_epoch = pd.concat(cycle2_df_epoch)

    cycle1_df_step["Acc/Train_VQA"] = cycle1_df_step["Acc/Train_VQA"] * 100
    cycle2_df_step["Acc/Train_VQA"] = cycle2_df_step["Acc/Train_VQA"] * 100
    cycle1_df_epoch["Acc/Val_VQA"] = cycle1_df_epoch["Acc/Val_VQA"] * 100
    cycle2_df_epoch["Acc/Val_VQA"] = cycle2_df_epoch["Acc/Val_VQA"] * 100
    # Train vqa accuracy
    plt.figure(figsize=(8, 4))
    plt.suptitle("Train VQA accuracy")
    ax = plt.subplot(1, 2, 1)
  #  plt.ylim(0, 100)
    sns.lineplot(data=cycle1_df_step, x="step", y="Acc/Train_VQA", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")

    ax = plt.subplot(1, 2, 2)
  #  plt.ylim(0, 100)
    sns.lineplot(data=cycle2_df_step, x="step", y="Acc/Train_VQA", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")

    plt.savefig(f"{args.output_dir}/train_vqa_acc.png", bbox_inches="tight")
    plt.clf()

    # val vqa Accuracy

    plt.figure(figsize=(8, 4))
    plt.suptitle("Val VQA accuracy")
    ax = plt.subplot(1, 2, 1)
   # plt.ylim(0, 100)
    sns.lineplot(data=cycle1_df_epoch, x="step", y="Acc/Val_VQA", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")

    ax = plt.subplot(1, 2, 2)
   # plt.ylim(0, 100)
    sns.lineplot(data=cycle2_df_epoch, x="step", y="Acc/Val_VQA", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")

    plt.savefig(f"{args.output_dir}/val_vqa_acc.png", bbox_inches="tight")
    plt.clf()

    # FID
    plt.figure(figsize=(8, 4))
    plt.suptitle("Validation FID")
    ax = plt.subplot(1, 2, 1)
   # plt.ylim(0, 1)
    sns.lineplot(data=cycle1_df_epoch, x="step", y="FID/Val", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("FID")

    ax = plt.subplot(1, 2, 2)
  #  plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df_epoch, x="step", y="FID/Val", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("FID")

    plt.savefig(f"{args.output_dir}/val_fid.png", bbox_inches="tight")
    plt.clf()

    # Inception
    plt.figure(figsize=(8, 4))
    plt.suptitle("Validation Inception score")
    ax = plt.subplot(1, 2, 1)
   # plt.ylim(0, 1)
    sns.lineplot(data=cycle1_df_epoch, x="step", y="Inception/Val_Mean", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("FID")

    ax = plt.subplot(1, 2, 2)
  #  plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df_epoch, x="step", y="Inception/Val_Mean", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("FID")

    plt.savefig(f"{args.output_dir}/val_is.png", bbox_inches="tight")
    plt.clf()

    # Loss image consistency
    plt.figure(figsize=(8, 4))
    plt.suptitle("Loss image consistency")
    ax = plt.subplot(1, 2, 1)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle1_df_step, x="step", y="Loss/Image_Consistency", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    ax = plt.subplot(1, 2, 2)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df_step, x="step", y="Loss/Image_Consistency", ci="sd", hue=hue, linewidth=1,
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
    sns.lineplot(data=cycle1_df_step, x="step", y="Loss/Total", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    ax = plt.subplot(1, 2, 2)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df_step, x="step", y="Loss/Total", ci="sd", hue=hue, linewidth=1,
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
    sns.lineplot(data=cycle1_df_step, x="step", y="Loss/VQA", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    ax = plt.subplot(1, 2, 2)
    #plt.ylim(0, 1)
    sns.lineplot(data=cycle2_df_step, x="step", y="Loss/VQA", ci="sd", hue=hue, linewidth=1,
                 err_style=err_style).set_title(f"Cycle 2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(f"{args.output_dir}/loss_vqa.png", bbox_inches="tight")
    plt.clf()
