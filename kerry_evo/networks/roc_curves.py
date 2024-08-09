#!/usr/bin/env python

import fire
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from sklearn.metrics import auc
from statistics import mean

# def get_auc(path):
#     df = pd.read_csv(path)
#     return auc(df["fpr"], df["tpr"])


def run(dir, pattern, outpath, round=False):
    fig, ax = plt.subplots()

    def get_collection(domain):
        segments = []
        auc_ = []
        for i in os.listdir(dir):
            df = pd.read_csv(os.path.join(dir, i, pattern, domain))
            segments.append(list(zip(df["fpr"].to_list(), df["tpr"].to_list())))
            auc_.append(auc(df["fpr"], df["tpr"]))
        return LineCollection(segments), mean(auc_) 

    src_collection, src_auc = get_collection("source-roc.csv")
    tgt_collection, tgt_auc = get_collection("target-roc.csv")

    if round:
        mis_auc = f"{tgt_auc:0.1f}"
    else:
        mis_auc = f"{tgt_auc:0.2f}"
    ax.add_collection(src_collection).set(color="#1f77b4", label=f"No misspecification (Mean AUC: {src_auc:0.1f})")
    ax.add_collection(tgt_collection).set(color="#ff7f0e", label=f"Mispecification (Mean AUC: {mis_auc})")

    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # ax.set_title('ROC Curves')
    plt.legend(loc='lower right')
    fig.savefig(outpath)

if __name__ == "__main__":
    fire.Fire(run)

