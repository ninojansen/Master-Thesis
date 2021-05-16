
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
    parser.add_argument('--outdir', dest='output_dir', type=str,
                        default='/home/nino/Dropbox/Documents/Master/Thesis/Results/IG/plots')
    args = parser.parse_args()
    return args


def rename(x):
    if "vae" in x:
        return "vae"
    if "non_pretrained" in x:
        return "non_pretrained"
    else:
        return "pretrained"


def process_df(df):
    df = df.copy()
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
    df["run"] = df['run'].apply(lambda x: rename(x))
    return df


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    experiment_ids = {
        "BOW": "xPdJfL8xR9yl34OxZkEH7A", "SBERT_Full": "MukTeaFPTY6oDQWcDEVUrA",
        "SBERT_Reduced": "EIVQmTY7TxORRENVVcacdA", "PHOC_Full": "cPmSj5blQVy55R2Up4XVsA",
        "PHOC_Reduced": "BZG23jtbSG6wCDu1Wflngg"}

    metrics_ig = ['Acc/Fake', 'Acc/Real', 'FID/Val', 'Inception/Val_Mean',
                  'Loss/Discriminator', 'Loss/Generator']

    metrics_vae = ['Loss/KL', 'Loss/Recon', 'Loss/VAE_epoch']

    non_pretrained_dfs = []
    pretrained_dfs = []
    vae_dfs = []

    skip_full = True

    if skip_full:
        prefix = "ig_reduced"
    else:
        prefix = "ig_full"

    for key, value in experiment_ids.items():
        if skip_full and "full" in key.lower():
            continue
        experiment = tb.data.experimental.ExperimentFromDev(value)
        df = experiment.get_scalars()
        runs = df["run"].unique()
        # print(df["tag"].unique())
        vae = process_df(df[df["tag"].isin(metrics_vae) == True])
        vae["run"] = vae['run'].apply(lambda x: key)
        ig_df = process_df(df[df["tag"].isin(metrics_ig) == True])

        non_pretrained = ig_df[ig_df["run"] == "non_pretrained"]
        non_pretrained["run"] = non_pretrained['run'].apply(lambda x: key)
        pretrained = ig_df[ig_df["run"] == "non_pretrained"]
        pretrained["run"] = pretrained['run'].apply(lambda x: key)

        non_pretrained_dfs.append(non_pretrained)
        pretrained_dfs.append(pretrained)
        vae_dfs.append(vae)
        print()

        # Drop full runs

    non_pretrained_df = pd.concat(non_pretrained_dfs)
    pretrained_df = pd.concat(pretrained_dfs)
    vae_df = pd.concat(vae_dfs)

    non_pretrained_df.rename(columns={"run": "Text embedding"}, inplace=True)
    pretrained_df.rename(columns={"run": "Text embedding"}, inplace=True)
    vae_df.rename(columns={"run": "Text embedding"}, inplace=True)

    #hue = df["run"].apply(lambda x: x)

    plt.figure(figsize=(16, 6))
    plt.suptitle("Image Generation")
    ax = plt.subplot(2, 2, 1)
    # plt.ylim(0, 1)
    # ax.xaxis.set_major_locator(ticker.AutoLocator())
    sns.lineplot(data=non_pretrained_df, x="step", y="Acc/Fake",
                 hue="Text embedding", ci=None,).set_title(f"Fake image Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.ylim(0, 1)
    ax = plt.subplot(2, 2, 2)
    sns.lineplot(data=non_pretrained_df, x="step", y="Acc/Real",
                 hue="Text embedding", ci=None).set_title(f"Real image Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    ax = plt.subplot(2, 2, 3)
    sns.lineplot(data=non_pretrained_df, x="step", y="Loss/Discriminator",
                 hue="Text embedding", ci=None).set_title(f"Loss discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    ax = plt.subplot(2, 2, 4)
    sns.lineplot(data=non_pretrained_df, x="step", y="Loss/Generator",
                 hue="Text embedding", ci=None).set_title(f"Loss generator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

   # plt.show()
    plt.savefig(f"{args.output_dir}/{prefix}_acc_loss_plot.png", bbox_inches='tight')

    plt.figure(figsize=(16, 6))
    plt.suptitle("Pretrained Image Generation")
    ax = plt.subplot(1, 2, 1)
    # plt.ylim(0, 1)
    # ax.xaxis.set_major_locator(ticker.AutoLocator())
    sns.lineplot(data=pretrained_df, x="step", y="FID/Val", hue="Text embedding",
                 ci=None).set_title(f"Validation Fréchet inception distance")
    plt.xlabel("Epoch")
    plt.ylabel("Fréchet inception distance")
    # plt.ylim(0, 1)
    ax = plt.subplot(1, 2, 2)
    sns.lineplot(data=pretrained_df, x="step", y="Inception/Val_Mean", hue="Text embedding",
                 ci=None).set_title(f"Validation inception score")
    plt.xlabel("Epoch")
    plt.ylabel("Inception score")

   # plt.show()
    plt.savefig(f"{args.output_dir}/{prefix}_val_plot.png", bbox_inches='tight')

    plt.figure(figsize=(16, 6))
    plt.suptitle("Varitional auto-encoder")
    ax = plt.subplot(1, 2, 1)
    # plt.ylim(0, 1)
    # ax.xaxis.set_major_locator(ticker.AutoLocator())
    sns.lineplot(data=vae_df, x="step", y="Loss/KL", hue="Text embedding",
                 ci=None).set_title(f"Kullback–Leibler divergence loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.ylim(0, 1)
    ax = plt.subplot(1, 2, 2)
    sns.lineplot(data=vae_df, x="step", y="Loss/Recon", hue="Text embedding",
                 ci=None).set_title(f"Reconstruction loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    ax = plt.subplot(1, 3, 3)
    sns.lineplot(data=vae_df, x="step", y="Loss/VAE_epoch", hue="Text embedding",
                 ci=None).set_title(f"Full loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
  #  plt.show()
    plt.savefig(f"{args.output_dir}/f{prefix}_vae_plot.png", bbox_inches='tight')
