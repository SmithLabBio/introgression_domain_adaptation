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


def plot(cm, outpath):
    labels = ["No Migration", "Migration"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, ax=ax, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, 
                yticklabels=labels, cbar=False, annot_kws={"size":20}, square=True)#, linewidth=1, linecolor="black")
    sns.despine(ax=ax, top=False, right=False, left=False, bottom=False)
    ax.tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    fig.savefig(outpath, bbox_inches="tight")
    print(outpath)

def run(input_dir, test_dir_name, prefix):
    def get_cm(dir, domain):
        cms = []
        for i in os.listdir(dir):
            path = f"{dir}/{i}/{test_dir_name}/{domain}-cm.csv"
            cm = np.loadtxt(path, dtype=float, delimiter=',')
            cms.append(cm)
        summed = np.array(cms).sum(axis=0)
        return summed / summed.sum(axis=1)

    source_cm = get_cm(input_dir, "source")
    target_cm = get_cm(input_dir, "target")

    print("Source CM:")
    print(source_cm)
    print("Target CM")
    print(target_cm)

    plot(source_cm, prefix + ".src.cm.pdf")
    plot(target_cm, prefix + ".tgt.cm.pdf")


if __name__ == "__main__":
    fire.Fire(run)
