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

plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13 
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 15 
plt.rcParams["figure.titlesize"] = 18

src_color = "#1f78b4"
tgt_color = "#d95f02"

def run(input, outpath):
    source_arrs = []
    target_arrs = []
    for i in os.listdir(input):
        data = np.load(f"{input}/{i}/test-epoch-50/latent-space.npz")
        s = data["source"]
        t = data["target"]
        source_arrs.append(s)
        target_arrs.append(t)
        
        c = np.concatenate((s, t))

        out = f"{outpath}.pca.{i}.pdf"
        pca = PCA().fit_transform(c)
        fig, ax = plt.subplots()
        ax.plot(pca[:len(s) , 0], pca[:len(s),  1], '.', c=src_color, label="Source", alpha=1.0, markersize=5, zorder=2)
        ax.plot(pca[ len(s):, 0], pca[ len(s):, 1], '.', c=tgt_color, label="Target", alpha=1.0, markersize=5, zorder=1)
        ax.legend(loc='upper right', markerscale=2.5)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_box_aspect(1)
        fig.savefig(out, bbox_inches="tight")


    source = np.concatenate(source_arrs) 
    target = np.concatenate(target_arrs)
    combined = np.concatenate((source, target))

    out = f"{outpath}.pca.pdf"
    pca = PCA().fit_transform(combined)
    fig1, ax1 = plt.subplots()
    ax1.plot(pca[:len(source) , 0], pca[:len(source),  1], '.', c=src_color, label="Source", alpha=1.0, markersize=5, zorder=2)
    ax1.plot(pca[ len(source):, 0], pca[ len(source):, 1], '.', c=tgt_color, label="Target", alpha=1.0, markersize=5, zorder=1)
    ax1.legend(loc='upper right', markerscale=2.5)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.set_box_aspect(1)
    fig1.savefig(out, bbox_inches="tight")
    print(out)



    ## Code to plot tsne
    # outpath2 = f"{outpath}.tsne.pdf"
    # tsne = TSNE(2).fit_transform(combined)
    # fig2, ax2 = plt.subplots()
    # ax2.plot(tsne[:len(source), 0], tsne[:len(source), 1], '.', label="Source", color=src_color)
    # ax2.plot(tsne[len(source):, 0], tsne[len(source):, 1], '.', label="Target", color=tgt_color)
    # ax2.legend(loc='lower right')
    # ax2.spines[['right', 'top']].set_visible(False)
    # ax2.set_box_aspect(1)
    # fig2.savefig(outpath2, bbox_inches="tight")
    # print(outpath2)


## Code to plot replicates individually
    # data = np.load(input)
    # source = data["source"]
    # target = data["target"]
    # combined = np.concatenate((source, target))

    # outpath1 = f"{outpath}.pca.pdf"
    # pca = PCA().fit_transform(combined)
    # fig1, ax1 = plt.subplots()
    # ax1.plot(pca[:len(source) , 0], pca[:len(source),  1], '.', c=src_color, label="Source", alpha=0.5, zorder=2)
    # ax1.plot(pca[ len(source):, 0], pca[ len(source):, 1], '.', c=tgt_color, label="Target", alpha=0.5, zorder=1)
    # ax1.legend(loc='lower right')
    # fig1.savefig(outpath1)
    # print(outpath1)

    # outpath2 = f"{outpath}.tsne.pdf"
    # tsne = TSNE(2).fit_transform(combined)
    # fig2, ax2 = plt.subplots()
    # ax2.plot(tsne[:len(source), 0], tsne[:len(source), 1], '.', label="Source", color=src_color)
    # ax2.plot(tsne[len(source):, 0], tsne[len(source):, 1], '.', label="Target", color=tgt_color)
    # ax2.legend(loc='lower right')
    # fig2.savefig(outpath2)
    # print(outpath2)


if __name__ == "__main__":
    fire.Fire(run)
