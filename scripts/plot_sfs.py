#!/usr/bin/env python

import fire
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from sklearn.metrics import auc
from statistics import mean
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13 
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 15 
plt.rcParams["figure.titlesize"] = 18

src_color_m0 = "#264653"
src_color_m1 = "#2A9D8F"
tgt_color_m0 = "#E76F51"
tgt_color_m1 = "#E9C46A"


def run(sourcepath, targetpath, outpath):

    # read data
    src = np.load(f"{sourcepath}-norm.npz")
    src_data = src["x"]
    tgt = np.load(f"{targetpath}-norm.npz")
    tgt_data = tgt["x"]

    # split based on labels
    src_mask_0 = src["labels"] == 0
    src_mask_1 = src["labels"] == 1
    tgt_mask_0 = tgt["labels"] == 0
    tgt_mask_1 = tgt["labels"] == 1

    src_0 = src_data[src_mask_0]
    src_1 = src_data[src_mask_1]

    tgt_0 = tgt_data[tgt_mask_0]
    tgt_1 = tgt_data[tgt_mask_1]

    # average
    src_0_meansfs = np.mean(src_0, axis=0).squeeze()
    src_1_meansfs = np.mean(src_1, axis=0).squeeze()
    tgt_0_meansfs = np.mean(tgt_0, axis=0).squeeze()
    tgt_1_meansfs = np.mean(tgt_1, axis=0).squeeze()

    # out path 
    out = f"{outpath}.sfs.v2.pdf"

    # Plot SFS
    fig, axs = plt.subplots(2, 2)
    sns.heatmap(data=src_0_meansfs, ax=axs[0,0], norm=LogNorm(), vmin=0, vmax=1, cbar=False, yticklabels=False, xticklabels=False, square=True)
    sns.heatmap(data=src_1_meansfs, ax=axs[0,1], norm=LogNorm(), vmin=0, vmax=1, cbar=False, yticklabels=False, xticklabels=False, square=True)
    sns.heatmap(data=tgt_0_meansfs, ax=axs[1,0], norm=LogNorm(), vmin=0, vmax=1, cbar=False, yticklabels=False, xticklabels=False, square=True)
    sns.heatmap(data=tgt_1_meansfs, ax=axs[1,1], norm=LogNorm(), vmin=0, vmax=1, cbar=False, yticklabels=False, xticklabels=False, square=True)

    axs[0,0].set_title("Source No Migration", fontsize=10)
    axs[0,1].set_title("Source Migration", fontsize=10)
    axs[1,0].set_title("Target No Migration", fontsize=10)
    axs[1,1].set_title("Target Migration", fontsize=10)

    for i, ax in enumerate(fig.get_axes()):
        ax.text(-0.05, 1.02, f"{chr(97 + i)})", transform=ax.transAxes, size=8, weight="bold")

    plt.tight_layout(rect=[0, 0, .9, 1])
    cbar_ax = fig.add_axes([0.90, 0.3, 0.02, 0.41])
    cbar_ax.tick_params(labelsize=8) 
    fig.colorbar(axs[1,1].collections[0], cax=cbar_ax)
    fig.savefig(out)

    print(out)

if __name__ == "__main__":
    fire.Fire(run)
