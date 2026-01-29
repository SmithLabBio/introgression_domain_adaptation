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
all_s_colors = []
all_t_colors = []

def run(input, outpath):
    source_arrs = []
    target_arrs = []
    for i in os.listdir(input):
        if not os.path.isdir(f"{input}/{i}"):
            continue
        data = np.load(f"{input}/{i}/test-epoch-50/latent-space.npz")
        s = data["source"]
        t = data["target"]
        source_arrs.append(s)
        target_arrs.append(t)
        c = np.concatenate((s, t))
        try:
            s_labels = pd.read_csv(f"{input}/{i}/test-epoch-50/source-predictions.csv")
            t_labels = pd.read_csv(f"{input}/{i}/test-epoch-50/target-predictions.csv")
            s_colors = [src_color_m0 if x==0 else src_color_m1 for x in s_labels["labels"]]
            t_colors = [tgt_color_m0 if x==0 else tgt_color_m1 for x in t_labels["labels"]]
        except:
            s_labels = [0] * (len(s) // 2) + [1] * (len(s) // 2)
            t_labels = [0] * (len(t) // 2) + [1] * (len(t) // 2)
            s_colors = [src_color_m0 if x==0 else src_color_m1 for x in s_labels]
            t_colors = [tgt_color_m0 if x==0 else tgt_color_m1 for x in t_labels]
        all_s_colors.extend(s_colors)
        all_t_colors.extend(t_colors)
        out = f"{outpath}.pca.{i}_v2.pdf"
        pca = PCA().fit_transform(c)
        fig, ax = plt.subplots()
        ax.scatter(pca[:len(s), 0], pca[:len(s), 1], c=s_colors, alpha=1.0, s=5, zorder=2)
        ax.scatter(pca[len(s):, 0], pca[len(s):, 1], c=t_colors, alpha=1.0, s=5, zorder=1)
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
        ax.set_box_aspect(1)
        fig.savefig(out, bbox_inches="tight")


    source = np.concatenate(source_arrs) 
    target = np.concatenate(target_arrs)
    combined = np.concatenate((source, target))

    out = f"{outpath}.pca.v2.pdf"
    pca = PCA().fit_transform(combined)
    fig1, ax1 = plt.subplots()
    ax1.scatter(pca[:len(source), 0], pca[:len(source), 1], c=all_s_colors, s=5, alpha=0.5, zorder=2)
    ax1.scatter(pca[len(source):, 0], pca[len(source):, 1], c=all_t_colors, s=5, alpha=0.5, zorder=1)
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
    ax1.legend(handles=legend_elements, frameon=False, fontsize=12)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.set_box_aspect(1)
    fig1.savefig(out, bbox_inches="tight")
    plt.close(fig1)
    print(out)

if __name__ == "__main__":
    fire.Fire(run)
