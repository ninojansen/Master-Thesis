
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
from tqdm import tqdm
import math


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


def process_vae_df(df):
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
    df["step"] = df['step'].apply(lambda x: math.floor(x / df['step'][0]))
    df["run"] = df['run'].apply(lambda x: rename(x))
    return df


def process_ig_df(df, key, smoothing=5):
    ig_df = df.copy()
    ig_df = ig_df.pivot_table(values=("value"), columns="tag", index=["run", "step"])
    # `reset_index()` removes the MultiIndex structure of the pivoted
    # DataFrame. Before the call, the DataFrame consits of two levels
    # of index: "run" and "step". After the call, the index become a
    # single range index (e.g,. `dataframe[:2]` works).
    ig_df = ig_df.reset_index()
    ig_df.columns.name = None
    # Remove the columns name "tag".
    ig_df.columns.names = [None for name in ig_df.columns.names]
    # Change step to epoch
    # df["step"] = df['step'].apply(lambda x: x % df['step'][0])
    ig_df["run"] = ig_df['run'].apply(lambda x: rename(x))

    non_pretrained = ig_df[ig_df["run"] == "non_pretrained"].copy()
    non_pretrained["step"] = non_pretrained['step'].apply(lambda x: math.floor(x / non_pretrained["step"][0]))

    # non_pretrained["run"] = non_pretrained['run'].apply(lambda x: key)
    pretrained = ig_df[ig_df["run"] == "pretrained"].copy()
    pretrained["step"] = pretrained['step'].apply(lambda x: math.floor(x / pretrained["step"][400]))

    # pretrained["run"] = pretrained['run'].apply(lambda x: key)

    non_pretrained = non_pretrained.groupby(np.arange(len(non_pretrained)) // smoothing).mean()
   # non_pretrained["step"] = non_pretrained['step'].apply(lambda x: int(x * smoothing))
    non_pretrained["run"] = key

    pretrained = pretrained.groupby(np.arange(len(pretrained)) // smoothing).mean()
   # pretrained["step"] = pretrained['step'].apply(lambda x: int(x * smoothing))
    pretrained["run"] = key

    return non_pretrained, pretrained


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    non_pretrained_dir = os.path.join(args.output_dir, "non_pretrained")
    pretrained_dir = os.path.join(args.output_dir, "pretrained")
    vae_dir = os.path.join(args.output_dir, "vae")

    os.makedirs(non_pretrained_dir, exist_ok=True)
    os.makedirs(pretrained_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)

    experiment_ids = {
        "BOW": ["CdzRbcyjQoG0wginYoLPrw", "L3BKG5xuRbO2cxDWE9jXaQ", "iAjOHwK1T0OuAR8VZ8gvNA"],
        "SBERT Full": ["6v2bS0hBQlOd1b5bY2pkCw", "jUpQdNJQQgGYA2WDmbZrNg", "zjhnYymeQdW2up1XWYzMUA"],
        "SBERT Reduced": ["AT7vpap7Qnmtr7IW91ASZg", "0FXnCOZzS7mwP09PZA6PyA", "xEGnHR91RXa4rlKIVxNvxQ"],
        "PHOC Full": ["7AmLWROCQEqWEAmSYuVdtw", "BnXc0qqBR3moO50t9BUnTQ", "lTWszxUaT7K75I77VvCYcw"],
        "PHOC Reduced": ["4xvPDYP7Sba61U1lvlsmrg", "zdpOuZbNTAKKYAiN6N7OvA", "0aT3qJ8MR2yPnWyDfwcK1Q"]}

    metrics_ig = ['Acc/Fake', 'Acc/Real', 'FID/Val', 'Inception/Val_Mean',
                  'Loss/Discriminator', 'Loss/Generator']

    metrics_vae = ['Loss/KL', 'Loss/Recon', 'Loss/VAE_epoch']

    smoothing_value = 5

    non_pretrained_dfs = []
    pretrained_dfs = []
    vae_dfs = []

    skip_full = False

    if skip_full:
        prefix = "ig_reduced"
    else:
        prefix = "ig_full"
    sns.set_theme()
    sns.set_context("paper")
    for key, values in tqdm(experiment_ids.items()):
        for value in values:
            if skip_full and "full" in key.lower():
                continue
            experiment = tb.data.experimental.ExperimentFromDev(value)
            df = experiment.get_scalars()
            runs = df["run"].unique()
            # print(df["tag"].unique())
            vae = process_vae_df(df[df["tag"].isin(metrics_vae) == True])
            vae["run"] = vae['run'].apply(lambda x: key)

            non_pretrained, pretrained = process_ig_df(
                df[df["tag"].isin(metrics_ig) == True],
                key, smoothing=smoothing_value)

            non_pretrained_dfs.append(non_pretrained)
            pretrained_dfs.append(pretrained)
            vae_dfs.append(vae)

        # Drop full runs

    non_pretrained_df = pd.concat(non_pretrained_dfs)
    pretrained_df = pd.concat(pretrained_dfs)
    vae_df = pd.concat(vae_dfs)

    non_pretrained_df.rename(columns={"run": "text embedding"}, inplace=True)
    pretrained_df.rename(columns={"run": "text embedding"}, inplace=True)
    vae_df.rename(columns={"run": "text embedding"}, inplace=True)

    # hue = df["run"].apply(lambda x: x)
    hue = "text embedding"
    err_style = "band"

    # Fake accuracy plots
    plt.figure(figsize=(5, 4), dpi=300)
    plt.ylim(0, 1)
    sns.lineplot(data=non_pretrained_df, x="step", y="Acc/Fake", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Fake Accuracy")
    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")
    plt.savefig(f"{non_pretrained_dir}/fake_accuracy.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(5, 4), dpi=300)
    plt.ylim(0, 1)
    sns.lineplot(data=pretrained_df, x="step", y="Acc/Fake", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Fake Accuracy")
    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")
    plt.savefig(f"{pretrained_dir}/fake_accuracy.png", bbox_inches="tight")
    plt.clf()

    # Real accuracy plots
    plt.figure(figsize=(5, 4), dpi=300)
    plt.ylim(0, 1)
    sns.lineplot(data=non_pretrained_df, x="step", y="Acc/Real", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Real Accuracy")
    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")
    plt.savefig(f"{non_pretrained_dir}/real_accuracy.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(5, 4), dpi=300)
    plt.ylim(0, 1)
    sns.lineplot(data=pretrained_df, x="step", y="Acc/Real", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Real Accuracy")
    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")
    plt.savefig(f"{pretrained_dir}/real_accuracy.png", bbox_inches="tight")
    plt.clf()

   # Loss Discriminator

    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=non_pretrained_df, x="step", y="Loss/Discriminator", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Loss discriminator")
    plt.xlabel("Epoch")

    plt.ylabel("Loss")
    plt.savefig(f"{non_pretrained_dir}/loss_discriminator.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=pretrained_df, x="step", y="Loss/Discriminator", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Loss discriminator")
    plt.xlabel("Epoch")

    plt.ylabel("Loss")
    plt.savefig(f"{pretrained_dir}/loss_discriminator.png", bbox_inches="tight")
    plt.clf()
   # Loss Generator
    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=non_pretrained_df, x="step", y="Loss/Generator", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Loss generator")
    plt.xlabel("Epoch")

    plt.ylabel("Loss")
    plt.savefig(f"{non_pretrained_dir}/loss_generator.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=pretrained_df, x="step", y="Loss/Generator", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Loss generator")
    plt.xlabel("Epoch")

    plt.ylabel("Loss")
    plt.savefig(f"{pretrained_dir}/loss_generator.png", bbox_inches="tight")
    plt.clf()

    # Val FID
    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=non_pretrained_df, x="step", y="FID/Val", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Validation Fréchet inception distance")
    plt.xlabel("Epoch")

    plt.ylabel("FID")
    plt.savefig(f"{non_pretrained_dir}/fid_val.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=pretrained_df, x="step", y="FID/Val", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Validation Fréchet inception distance")
    plt.xlabel("Epoch")

    plt.ylabel("FID")
    plt.savefig(f"{pretrained_dir}/fid_val.png", bbox_inches="tight")
    plt.clf()
   # Val IS
    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=non_pretrained_df, x="step", y="Inception/Val_Mean", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Validation inception score")
    plt.xlabel("Epoch")

    plt.ylabel("Inception score")
    plt.savefig(f"{non_pretrained_dir}/is_val.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=pretrained_df, x="step", y="Inception/Val_Mean", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Validation inception score")
    plt.xlabel("Epoch")

    plt.ylabel("Inception score")
    plt.savefig(f"{pretrained_dir}/is_val.png", bbox_inches="tight")
    plt.clf()

    # VAE
    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=vae_df, x="step", y="Loss/KL", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Loss Kullback–Leibler")
    plt.xlabel("Epoch")

    plt.ylabel("Loss")
    plt.savefig(f"{vae_dir}/loss_kl.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=vae_df, x="step", y="Loss/Recon", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Loss Reconstruction")
    plt.xlabel("Epoch")

    plt.ylabel("Loss")
    plt.savefig(f"{vae_dir}/loss_recon.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=vae_df, x="step", y="Loss/VAE_epoch", hue=hue, ci="sd",
                 err_style=err_style, linewidth=1).set_title(f"Loss Total")
    plt.xlabel("Epoch")

    plt.ylabel("Loss")
    plt.savefig(f"{vae_dir}/loss_vae.png", bbox_inches="tight")
    plt.clf()


#     plt.figure(figsize=(16, 6))
#     plt.suptitle("Image Generation")
#     ax = plt.subplot(2, 2, 1)
#     # plt.ylim(0, 1)
#     # ax.xaxis.set_major_locator(ticker.AutoLocator())
#     sns.lineplot(data=non_pretrained_df, x="step", y="Acc/Fake",
#                  hue="Text embedding", ci=None,).set_title(f"Fake image Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     # plt.ylim(0, 1)
#     ax = plt.subplot(2, 2, 2)
#     sns.lineplot(data=non_pretrained_df, x="step", y="Acc/Real",
#                  hue="Text embedding", ci=None).set_title(f"Real image Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")

#     ax = plt.subplot(2, 2, 3)
#     sns.lineplot(data=non_pretrained_df, x="step", y="Loss/Discriminator",
#                  hue="Text embedding", ci=None).set_title(f"Loss discriminator")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")

#     ax = plt.subplot(2, 2, 4)
#     sns.lineplot(data=non_pretrained_df, x="step", y="Loss/Generator",
#                  hue="Text embedding", ci=None).set_title(f"Loss generator")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")

#    # plt.show()
#     plt.savefig(f"{args.output_dir}/{prefix}_acc_loss_plot.png", bbox_inches='tight')

#     plt.figure(figsize=(16, 6))
#     plt.suptitle("Pretrained Image Generation")
#     ax = plt.subplot(1, 2, 1)
#     # plt.ylim(0, 1)
#     # ax.xaxis.set_major_locator(ticker.AutoLocator())
#     sns.lineplot(data=pretrained_df, x="step", y="FID/Val", hue="Text embedding",
#                  ci=None).set_title(f"Validation Fréchet inception distance")
#     plt.xlabel("Epoch")
#     plt.ylabel("Fréchet inception distance")
#     # plt.ylim(0, 1)
#     ax = plt.subplot(1, 2, 2)
#     sns.lineplot(data=pretrained_df, x="step", y="Inception/Val_Mean", hue="Text embedding",
#                  ci=None).set_title(f"Validation inception score")
#     plt.xlabel("Epoch")
#     plt.ylabel("Inception score")

#    # plt.show()
#     plt.savefig(f"{args.output_dir}/{prefix}_val_plot.png", bbox_inches='tight')

#     plt.figure(figsize=(16, 6))
#     plt.suptitle("Varitional auto-encoder")
#     ax = plt.subplot(1, 2, 1)
#     # plt.ylim(0, 1)
#     # ax.xaxis.set_major_locator(ticker.AutoLocator())
#     sns.lineplot(data=vae_df, x="step", y="Loss/KL", hue="Text embedding",
#                  ci=None).set_title(f"Kullback–Leibler divergence loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     # plt.ylim(0, 1)
#     ax = plt.subplot(1, 2, 2)
#     sns.lineplot(data=vae_df, x="step", y="Loss/Recon", hue="Text embedding",
#                  ci=None).set_title(f"Reconstruction loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")

#     ax = plt.subplot(1, 3, 3)
#     sns.lineplot(data=vae_df, x="step", y="Loss/VAE_epoch", hue="Text embedding",
#                  ci=None).set_title(f"Full loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#   #  plt.show()
#     plt.savefig(f"{args.output_dir}/f{prefix}_vae_plot.png", bbox_inches='tight')
