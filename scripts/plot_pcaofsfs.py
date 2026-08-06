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
    all_data = np.concatenate((src_data, tgt_data))

    # flatten
    all_data_flat = all_data.reshape(all_data.shape[0], -1)

    # out path 
    out = f"{outpath}.sfs.pca.v2.pdf"

    # PCA
    pca = PCA().fit_transform(all_data_flat)

    #plot
    s_colors = [src_color_m0 if x==0 else src_color_m1 for x in src["labels"]]
    t_colors = [tgt_color_m0 if x==0 else tgt_color_m1 for x in tgt["labels"]]
    fig, ax = plt.subplots()
    ax.scatter(pca[:len(src_data), 0], pca[:len(src_data), 1], c=s_colors, alpha=1.0, s=5, zorder=2)
    ax.scatter(pca[len(src_data):, 0], pca[len(src_data):, 1], c=t_colors, alpha=1.0, s=5, zorder=1)
    legend_elements = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=src_color_m0, markersize=6,
               label='Source-No Migration'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=src_color_m1, markersize=6,
               label='Source-Migration'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=tgt_color_m0, markersize=6,
               label='Target-No Migration'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=tgt_color_m1, markersize=6,
               label='Target-Migration'),
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=12)
    ax.spines[['right', 'top']].set_visible(False)
    ax.text(-0.20, 1.02, "e)", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.set_box_aspect(1)
    fig.savefig(out, bbox_inches="tight")


    print(out)

if __name__ == "__main__":
    fire.Fire(run)
