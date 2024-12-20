#!/usr/bin/env python

import fire
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from sklearn.metrics import auc
from statistics import mean
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10 
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 15 
plt.rcParams["figure.titlesize"] = 18
plt.rcParams["font.sans-serif"] = "Verdana"
# plt.rcParams["font.sans-serif"] = "Arial"


def rgb(r, g, b):
    w = 255
    return [r/w, g/w, b/w]

src_color = "#1f78b4"
tgt_color = "#d95f02"

# color1 = "black"
# color2 = "black"

# tgt_lstyle = ":"
tgt_lstyle = "solid"

def run(title, mispec_dir, adapted_dir, pattern, outpath, round=False):
    def get_collection(dir, domain):
        segments = []
        auc_ = []
        for i in os.listdir(dir):
            df = pd.read_csv(os.path.join(dir, i, pattern, domain))
            segments.append(list(zip(df["fpr"].to_list(), df["tpr"].to_list())))
            auc_.append(auc(df["fpr"], df["tpr"]))
        return LineCollection(segments), mean(auc_)  
    
    mispec_source_segments, mispec_source_auc = get_collection(mispec_dir, "source-roc.csv")
    mispec_target_segments, mispec_target_auc = get_collection(mispec_dir, "target-roc.csv")
    adapted_source_segments, adapted_source_auc = get_collection(adapted_dir, "source-roc.csv")
    adapted_target_segments, adapted_target_auc = get_collection(adapted_dir, "target-roc.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)


    ax1.add_collection(mispec_source_segments).set(color=src_color, zorder=2, 
            label=f"Source (Mean AUC: {mispec_source_auc:0.2f})")
    ax1.add_collection(mispec_target_segments).set(color=tgt_color, zorder=1, linestyle=tgt_lstyle,
            label=f"Target (Mean AUC: {mispec_target_auc:0.2f})")

    ax2.add_collection(adapted_source_segments).set(color=src_color, zorder=2, 
            label=f"Source (Mean AUC: {adapted_source_auc:0.2f})")
    ax2.add_collection(adapted_target_segments).set(color=tgt_color, zorder=1, linestyle=tgt_lstyle,
            label=f"Target (Mean AUC: {adapted_target_auc:0.2f})")

    def format(ax, title, letter, ylabel=True):
        ticks = [0, 0.25, 0.5, 0.75, 1]
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax.set_xlim([-0.03, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.spines[['right', 'top']].set_visible(False)
        ax.legend(loc='lower right')
        ax.set_xlabel('False Positive Rate')
        ax.xaxis.labelpad = 10
        ax.set_title(title, y=1.02)
        # ax.text(-0.13, 1.02, letter, size=15)
        if ylabel:
            ax.set_ylabel('True Positive Rate')


    format(ax1, 'No Domain Adaptation', "A)")
    format(ax2, 'Domain Adaptation', "B)", ylabel=False)

    fig.subplots_adjust(wspace=0.1)
    fig.suptitle(f"{title} Scenario ROC Curves", y=1.02)
    fig.savefig(outpath, bbox_inches="tight")

if __name__ == "__main__":
    fire.Fire(run)

