#!/usr/bin/env python

import fire
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
from statistics import mean

from sim_wrapper.simulator import Simulations
from simulations.secondary_contact import SecondaryContact
from simulations.secondary_contact_ghost2 import GhostSecondaryContact

plt.rcParams["axes.titlesize"] = 8
plt.rcParams["legend.fontsize"] = 2

def plot(dir, test_dir, source_test_data_sfs, source_test_data_ts, target_test_data_sfs, target_test_data_ts):
    def parse_tests(domain):
        labels = []
        predictions = []
        for rep in os.listdir(dir):
            dir_path = f"{dir}/{rep}/{test_dir}/"
            if os.path.isdir(dir_path):
                path = f"{dir_path}/{domain}-predictions.csv"
                df = pd.read_csv(path)
                labels = df["labels"]
                predictions.append(df["predictions"])
        return np.array(labels), np.array(predictions).T.mean(axis=1)

    source_labels, source_predictions = parse_tests("source") 
    target_labels, target_predictions = parse_tests("target") 

    source_tp = (source_predictions >  0.5) & (source_labels == 1)
    source_tn = (source_predictions <= 0.5) & (source_labels == 0)

    target_tp = (target_predictions >  0.5) & (target_labels == 1)
    target_fp = (target_predictions >  0.5) & (target_labels == 0)
    target_tn = (target_predictions <= 0.5) & (target_labels == 0)
    target_fn = (target_predictions <= 0.5) & (target_labels == 1)

    source_count_d = np.load(f"{source_test_data_sfs}.npz")["x"].squeeze(axis=-1)
    source_norm_d = np.load(f"{source_test_data_sfs}-norm.npz")["x"].squeeze(axis=-1)

    target_count_d = np.load(f"{target_test_data_sfs}.npz")["x"].squeeze(axis=-1)
    target_norm_d = np.load(f"{target_test_data_sfs}-norm.npz")["x"].squeeze(axis=-1)

    source_norm_tp_mean_sfs = np.mean(source_norm_d[source_tp], axis=0)
    source_norm_tn_mean_sfs = np.mean(source_norm_d[source_tn], axis=0)

    target_norm_tp_mean_sfs = np.mean(target_norm_d[target_tp], axis=0)
    target_norm_fp_mean_sfs = np.mean(target_norm_d[target_fp], axis=0)
    target_norm_tn_mean_sfs = np.mean(target_norm_d[target_tn], axis=0)
    target_norm_fn_mean_sfs = np.mean(target_norm_d[target_fn], axis=0)

    source_count_tp_mean_sfs = np.mean(source_count_d[source_tp], axis=0)
    source_count_tn_mean_sfs = np.mean(source_count_d[source_tn], axis=0)

    target_count_tp_mean_sfs = np.mean(target_count_d[target_tp], axis=0)
    target_count_fp_mean_sfs = np.mean(target_count_d[target_fp], axis=0)
    target_count_tn_mean_sfs = np.mean(target_count_d[target_tn], axis=0)
    target_count_fn_mean_sfs = np.mean(target_count_d[target_fn], axis=0)

    # Plot Normalized
    fig, axs = plt.subplots(2, 3)
    sns.heatmap(data=source_norm_tp_mean_sfs, ax=axs[0,0], norm=LogNorm(), vmin=0, vmax=1, cbar=False, yticklabels=False, xticklabels=False, square=True)
    sns.heatmap(data=target_norm_tp_mean_sfs, ax=axs[0,1], norm=LogNorm(), vmin=0, vmax=1, cbar=False, yticklabels=False, xticklabels=False, square=True)
    sns.heatmap(data=target_norm_fn_mean_sfs, ax=axs[0,2], norm=LogNorm(), vmin=0, vmax=1, cbar=False, yticklabels=False, xticklabels=False, square=True)
    sns.heatmap(data=source_norm_tn_mean_sfs, ax=axs[1,0], norm=LogNorm(), vmin=0, vmax=1, cbar=False, yticklabels=False, xticklabels=False, square=True)
    sns.heatmap(data=target_norm_tn_mean_sfs, ax=axs[1,1], norm=LogNorm(), vmin=0, vmax=1, cbar=False, yticklabels=False, xticklabels=False, square=True)
    sns.heatmap(data=target_norm_fp_mean_sfs, ax=axs[1,2], norm=LogNorm(), vmin=0, vmax=1, cbar=False, yticklabels=False, xticklabels=False, square=True)

    axs[0,0].set_title("Source True Positive")
    axs[0,1].set_title("Target True Positive")
    axs[0,2].set_title("Target False Negative")
    axs[1,0].set_title("Source True Negative")
    axs[1,1].set_title("Target True Negative")
    axs[1,2].set_title("Target False Positive")

    for i, ax in enumerate(fig.get_axes()):
        ax.text(-0.05, 1.02, f"{chr(97 + i)})", transform=ax.transAxes, size=8, weight="bold")

    plt.tight_layout(rect=[0, 0, .9, 1])
    cbar_ax = fig.add_axes([0.90, 0.3, 0.02, 0.41])
    cbar_ax.tick_params(labelsize=8) 
    fig.colorbar(axs[1,1].collections[0], cax=cbar_ax)
    # plt.tight_layout()
    fig.savefig(f"{dir}/misclassified_sfs_norm.pdf")

    # Plot Counts 
    fig, axs = plt.subplots(2, 3)
    sns.heatmap(data=source_count_tp_mean_sfs, ax=axs[0,0], vmin=0, vmax=6000, cbar=False, yticklabels=False, xticklabels=False, square=True, annot=True, fmt=".0f", annot_kws={"size":1})
    sns.heatmap(data=target_count_tp_mean_sfs, ax=axs[0,1], vmin=0, vmax=6000, cbar=False, yticklabels=False, xticklabels=False, square=True, annot=True, fmt=".0f", annot_kws={"size":1})
    sns.heatmap(data=target_count_fn_mean_sfs, ax=axs[0,2], vmin=0, vmax=6000, cbar=False, yticklabels=False, xticklabels=False, square=True, annot=True, fmt=".0f", annot_kws={"size":1})

    sns.heatmap(data=source_count_tn_mean_sfs, ax=axs[1,0], vmin=0, vmax=6000, cbar=False, yticklabels=False, xticklabels=False, square=True, annot=True, fmt=".0f", annot_kws={"size":1})
    sns.heatmap(data=target_count_tn_mean_sfs, ax=axs[1,1], vmin=0, vmax=6000, cbar=False, yticklabels=False, xticklabels=False, square=True, annot=True, fmt=".0f", annot_kws={"size":1})
    sns.heatmap(data=target_count_fp_mean_sfs, ax=axs[1,2], vmin=0, vmax=6000, cbar=False, yticklabels=False, xticklabels=False, square=True, annot=True, fmt=".0f", annot_kws={"size":1})

    axs[0,0].set_title("Source True Positive")
    axs[0,1].set_title("Target True Positive")
    axs[0,2].set_title("Target False Negative")
    axs[1,0].set_title("Source True Negative")
    axs[1,1].set_title("Target True Negative")
    axs[1,2].set_title("Target False Positive")

    for i, ax in enumerate(fig.get_axes()):
        ax.text(-0.05, 1.02, f"{chr(97 + i)})", transform=ax.transAxes, size=8, weight="bold")

    plt.tight_layout()
    fig.savefig(f"{dir}/misclassified_sfs_counts.pdf")


if __name__ == "__main__":
    fire.Fire(plot)