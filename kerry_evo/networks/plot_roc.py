#!/usr/bin/env python

import fire
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from sklearn.metrics import auc
from statistics import mean
from matplotlib.ticker import FormatStrFormatter
import math

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 15 
plt.rcParams["figure.titlesize"] = 18

def rd(x):
    return math.floor(x * 100) / 100

def rgb(r, g, b):
    w = 255
    return [r/w, g/w, b/w]

src_color = "#1f78b4"
tgt_color = "#d95f02"

# color1 = "black"
# color2 = "black"

# tgt_lstyle = ":"
tgt_lstyle = "solid"

def run(input_dir, test_dir_name, outpath):
    def get_collection(dir, domain, width):
        segments = []
        auc_ = []
        for i in os.listdir(dir):
            df = pd.read_csv(os.path.join(dir, i, test_dir_name, domain))
            segments.append(list(zip(df["fpr"].to_list(), df["tpr"].to_list())))
            auc_.append(auc(df["fpr"], df["tpr"]))
        return LineCollection(segments, linewidth=width), mean(auc_)  
    
    source_segments, source_auc = get_collection(input_dir, "source-roc.csv", 2)
    target_segments, target_auc = get_collection(input_dir, "target-roc.csv", 1.5)

    print("source auc: ", source_auc)
    print("target auc: ", target_auc)

    fig, ax = plt.subplots()

    ax.add_collection(source_segments).set(color=src_color, zorder=4, 
            label=f"Source\nMean AUC: {rd(source_auc):0.2f}")
    ax.add_collection(target_segments).set(color=tgt_color, zorder=3, linestyle=tgt_lstyle,
            label=f"Target\nMean AUC: {rd(target_auc):0.2f}")

    def format(ax):
        ticks = [0, 0.25, 0.5, 0.75, 1]
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', zorder=2)
        ax.set_xlim([-0.03, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.spines[['right', 'top']].set_visible(False)
        ax.legend(loc='lower right').set_zorder(1)
        ax.xaxis.labelpad = 10
        ax.set_aspect('equal')

    format(ax)

    fig.savefig(outpath, bbox_inches="tight")

if __name__ == "__main__":
    fire.Fire(run)

