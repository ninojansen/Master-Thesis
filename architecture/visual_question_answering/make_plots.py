
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
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument(
        '--path', dest='path', type=str,
        default="/home/nino/Dropbox/Documents/Master/Thesis/Models/VQA/vqa_experiment_final/top_attention/ef=bow_nhidden=256_lr=0.002")
    parser.add_argument('--outdir', dest='output_dir', type=str,
                        default='/home/nino/Dropbox/Documents/Master/Thesis/Results/vqa/plots')
    args = parser.parse_args()
    return args


def load_df(experiment_id, key):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    runs = df["run"].unique()
    # print(df["tag"].unique())

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
    # df = df.drop(df[(df.run == "ef=sbert_full_nhidden=256_lr=0.002") |
    #                 (df.run == "ef=phoc_full_nhidden=256_lr=0.002")].index)

    if key == "Vision only":
        df.rename(columns={"run": "network"}, inplace=True)
        hue = "network"
    else:
        df.rename(columns={"run": "text embedding"}, inplace=True)
        df["text embedding"] = df["text embedding"].apply(lambda x: x[3:x.index("nhidden") - 1])
        hue = "text embedding"
    return df, hue


if __name__ == "__main__":
    args = parse_args()

    # experiment_ids = {
    #     "Top Attention": "pGEQOLUNRzKQwZ1NwlpFAw", "Bottom + Top Attention": "o8wR6uk1QPygsPtV5RxA9A",
    #     "CNN": "hyzYeKOrT4uZEds8pqLODQ", "Pretrained": "tgjFqEKzRyOnYBwmLwhGew",
    #     "Language only": "u8PbrL4rT6SrtwPerQb8fA", "Vision only": "SO2pr03FRb65auQn5wNhPw"}

    experiment_ids = {
        "Top Attention": ["ocD15BvnQX259LB3wGxTlQ", "JTEKPNtMRhqzQCXJdlKu3g", "8nRXgNe0RNeBzM8ToQDupg"],
        "Bottom + Top Attention": ["6c95LLqrTnqyoWg369cN0g", "We8MmrAhRAmyBdwZB9b3Cg", "r1zGgDJjRnK0mtWtvckAsA"],
        "CNN": ["PyimvbORRVWdXnF44qIzBg", "KuD3ywaYTdaYaVYw16kDGQ", "7gaANjjYTZ2QkJrX2MWsdw"],
        "Pretrained": ["YMn2tgx1RvOzelcNT4wqRQ", "V08s1OQOQTaVUpecrPLksA", "rWNw72FPSaqCoiwl0ydjnw"],
        "Language only": ["NljrGxTlSNKHPejG6Jkh0w", "TvgMDY1oQoeBlYMkpzmLrA", "9Cg1i9CaSMerI42pWeuLCw"],
        "Vision only": ["y10yBlI3QZe7ulMcAqbztg", "qp8Di3PJQB60weeOPDVjVg", "2RGcQAenTliw6mL7cqwN0A"]}

    acc_dir = os.path.join(args.output_dir, "accuracy")
    loss_dir = os.path.join(args.output_dir, "accuracy")
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    # TODO
    # Start from zero
    # Make plots per cnn_type
    # Chnage y to epoch
    sns.set_theme()
    sns.set_context("paper")
    err_style = "band"

    metrics = ["Acc/Train", "Acc/Val", "Loss/CrossEntropy_epoch"]
    for key, values in tqdm(experiment_ids.items()):
        dfs = []
        for value in values:
            df, hue = load_df(value, key)
            dfs.append(df)

        df_total = pd.concat(dfs)

        # if key == "Vision only":
        #     df.rename(columns={"run": "Network"}, inplace=True)
        #     optimizer_validation = df["Network"]
        # else:
        #     df.rename(columns={"run": "Text Embedding"}, inplace=True)
        #     optimizer_validation = df["Text Embedding"].apply(lambda x: x[3:x.index("nhidden") - 1])

        plt.figure(figsize=(16, 4), dpi=300)
        plt.suptitle(key)
        ax = plt.subplot(1, 3, 1)
        plt.ylim(0, 1)
        sns.lineplot(data=df_total, x="step", y="Acc/Train", hue=hue, ci="sd", linewidth=1,
                     err_style=err_style).set_title(f"Training accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
       # plt.savefig(f"{acc_dir}/{key}_train.png")
      #  plt.clf()

        ax = plt.subplot(1, 3, 2)
        plt.ylim(0, 1)
        sns.lineplot(data=df_total, x="step", y="Acc/Val", hue=hue, ci="sd", linewidth=1,
                     err_style=err_style).set_title(f"Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
      #  plt.savefig(f"{acc_dir}/{key}_val.png")
     #   plt.clf()

        ax = plt.subplot(1, 3, 3)
        #plt.ylim(0, 1)
        sns.lineplot(data=df_total, x="step", y="Loss/CrossEntropy_epoch", hue=hue, ci="sd", linewidth=1,
                     err_style=err_style).set_title(f"Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
       # plt.savefig(f"{loss_dir}/{key}_loss.png")
      #  plt.clf()

        plt.savefig(f"{args.output_dir}/{key}_full_plot.png", bbox_inches="tight")
        plt.close()

        # plt.figure(figsize=(8, 6), dpi=120)
        # plt.ylim(0, 1)
        # sns.lineplot(data=df_total, x="step", y="Acc/Train", hue=hue, ci="sd", linewidth=1,
        #              err_style=err_style).set_title(f"Training accuracy")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.savefig(f"{acc_dir}/{key}_train.png")
        # plt.close()

        # plt.figure(figsize=(8, 6), dpi=120)
        # plt.ylim(0, 1)
        # sns.lineplot(data=df_total, x="step", y="Acc/Val", hue=hue, ci="sd", linewidth=1,
        #              err_style=err_style).set_title(f"Validation accuracy")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.savefig(f"{args.output_dir}/{key}_full_plot.png", bbox_inches="tight")
        # plt.close()

        # plt.figure(figsize=(8, 6), dpi=120)
        # #plt.ylim(0, 1)
        # sns.lineplot(data=df_total, x="step", y="Loss/CrossEntropy_epoch", hue=hue, ci="sd", linewidth=1,
        #              err_style=err_style).set_title(f"Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.savefig(f"{loss_dir}/{key}_loss.png")
        # plt.close()

    #    # plt.show()
    #     plt.figure(figsize=(16, 4), dpi=300)
    #     plt.suptitle(key)
    #     ax = plt.subplot(1, 3, 1)
    #     plt.ylim(0, 1)
    #   #  ax.xaxis.set_major_locator(ticker.AutoLocator())
    #     sns.lineplot(data=df, x="step", y="Acc/Train", ci="sd", hue=hue, linewidth=1,
    #                  err_style=err_style).set_title(f"Training accuracy")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Accuracy")

    #     plt.ylim(0, 1)
    #     ax = plt.subplot(1, 3, 2)
    #     sns.lineplot(data=df, x="step", y="Acc/Val", ci="sd", hue=hue, linewidth=1,
    #                  err_style=err_style).set_title(f"Validation accuracy")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Accuracy")
    #    # ax.xaxis.set_major_locator(ticker.AutoLocator())
    #     # plt.ylim(0, 1)
    #     ax = plt.subplot(1, 3, 3)
    #     sns.lineplot(data=df, x="step", y="Loss/CrossEntropy_epoch", ci="sd", linewidth=1,
    #                  hue=hue, err_style=err_style).set_title(f"Loss")
    #   #  ax.xaxis.set_major_locator(ticker.AutoLocator())
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")

    #     plt.savefig(f"{args.output_dir}/{key}_full_plot.png", bbox_inches="tight")
    #     plt.clf()
    #     # plt.show()
