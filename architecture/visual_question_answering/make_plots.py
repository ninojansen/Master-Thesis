
import os
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument(
        '--path', dest='path', type=str,
        default="/home/nino/Dropbox/Documents/Master/Thesis/Models/VQA/vqa_experiment_final/top_attention/ef=bow_nhidden=256_lr=0.002")
    parser.add_argument('--outdir', dest='output_dir', type=str,
                        default='/home/nino/Dropbox/Documents/Master/Thesis/Results/plots')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    experiment_ids = {
        "Top Attention": "pGEQOLUNRzKQwZ1NwlpFAw", "Bottom + Top Attention": "o8wR6uk1QPygsPtV5RxA9A",
        "CNN": "hyzYeKOrT4uZEds8pqLODQ", "Pretrained": "tgjFqEKzRyOnYBwmLwhGew",
        "Language only": "u8PbrL4rT6SrtwPerQb8fA", "Vision only": "SO2pr03FRb65auQn5wNhPw"}

    # TODO
    # Start from zero
    # Make plots per cnn_type
    # Chnage y to epoch

    metrics = ["Acc/Train", "Acc/Val", "Loss/CrossEntropy_epoch"]
    for key, value in experiment_ids.items():
        experiment = tb.data.experimental.ExperimentFromDev(value)
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
        df["step"] = df['step'].apply(lambda x: x % df['step'][0])
        df = df.drop(df[(df.run == "ef=sbert_full_nhidden=256_lr=0.002") |
                        (df.run == "ef=phoc_full_nhidden=256_lr=0.002")].index)
        if key == "Vision only":
            df.rename(columns={"run": "Network"}, inplace=True)
            optimizer_validation = df["Network"]
        else:
            df.rename(columns={"run": "Text Embedding"}, inplace=True)
            optimizer_validation = df["Text Embedding"].apply(lambda x: x[3:x.index("nhidden") - 1])

        plt.figure(figsize=(16, 6))
        plt.suptitle(key)
        ax = plt.subplot(1, 3, 1)
       # plt.ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        sns.lineplot(data=df, x="step", y="Acc/Train", hue=optimizer_validation).set_title(f"Training accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
       # plt.ylim(0, 1)
        ax = plt.subplot(1, 3, 2)
        sns.lineplot(data=df, x="step", y="Acc/Val", hue=optimizer_validation).set_title(f"Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        ax.xaxis.set_major_locator(ticker.AutoLocator())
       # plt.ylim(0, 1)
        ax = plt.subplot(1, 3, 3)
        sns.lineplot(data=df, x="step", y="Loss/CrossEntropy_epoch",
                     hue=optimizer_validation).set_title(f"Training Loss")
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.savefig(f"{args.output_dir}/{key}_acc_loss_plot.png")
      #  plt.show()
