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


def plot(type, domain, dir, test_dir, test_data_sfs, test_data_ts, out_prefix):
    labels = []
    predictions = []
    for rep in os.listdir(dir):
        dir_path = f"{dir}/{rep}/{test_dir}/"
        if os.path.isdir(dir_path):
            path = f"{dir_path}/{domain}-predictions.csv"
            df = pd.read_csv(path)
            labels = df["labels"]
            predictions.append(df["predictions"])
    labels = np.array(labels)
    predictions = np.array(predictions).T.mean(axis=1)

    tp = (predictions >  0.5) & (labels == 1)
    fp = (predictions >  0.5) & (labels == 0)
    tn = (predictions <= 0.5) & (labels == 0)
    fn = (predictions <= 0.5) & (labels == 1)

    count_d = np.load(f"{test_data_sfs}.npz")["x"].squeeze(axis=-1)
    norm_d = np.load(f"{test_data_sfs}-norm.npz")["x"].squeeze(axis=-1)

    norm_tp_mean_sfs = np.mean(norm_d[tp], axis=0)
    norm_fp_mean_sfs = np.mean(norm_d[fp], axis=0)
    norm_tn_mean_sfs = np.mean(norm_d[tn], axis=0)
    norm_fn_mean_sfs = np.mean(norm_d[fn], axis=0)

    count_tp_mean_sfs = np.mean(count_d[tp], axis=0)
    count_fp_mean_sfs = np.mean(count_d[fp], axis=0)
    count_tn_mean_sfs = np.mean(count_d[tn], axis=0)
    count_fn_mean_sfs = np.mean(count_d[fn], axis=0)

    def norm_heat(data, out): 
        fig, ax = plt.subplots()
        sns.heatmap(data, ax=ax, norm=LogNorm(vmin=0.0000095, vmax=1), cbar=True, yticklabels=False, 
                    xticklabels=False, square=True)
        ax.tick_params(left=False, bottom=False)
        sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2)
        plt.savefig(f"{out_prefix}/{domain}-{out}.pdf")
    
    norm_heat(norm_tp_mean_sfs, "norm_true_pos_sfs")
    norm_heat(norm_fp_mean_sfs, "norm_false_pos_sfs")
    norm_heat(norm_tn_mean_sfs, "norm_true_neg_sfs")
    norm_heat(norm_fn_mean_sfs, "norm_false_neg_sfs")
    
    def count_heat(data, out): 
        fig, ax = plt.subplots()
        sns.heatmap(data, ax=ax, cbar=True, yticklabels=False, 
                    xticklabels=False, square=True, 
                    annot=True,
                    fmt=".0f",
                    annot_kws={'size': 3, 'ha': 'center', 'va': 'center'} )

        ax.tick_params(left=False, bottom=False)
        sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2)
        plt.savefig(f"{out_prefix}/{domain}-{out}.pdf")

    count_heat(count_tp_mean_sfs, "count_true_pos_sfs")
    count_heat(count_fp_mean_sfs, "count_false_pos_sfs")
    count_heat(count_tn_mean_sfs, "count_true_neg_sfs")
    count_heat(count_fn_mean_sfs, "count_false_neg_sfs")

    scenario = eval(type)
    with open(test_data_ts, "r") as fh:
        json_data = fh.read()
    sims = Simulations[scenario, scenario._data_class].model_validate_json(json_data)

    tp_ts = [s for s, b in zip(sims, tp) if b]
    fp_ts = [s for s, b in zip(sims, fp) if b]
    tn_ts = [s for s, b in zip(sims, tn) if b]
    fn_ts = [s for s, b in zip(sims, fn) if b]

    def hist(param): 
        plt.figure()
        sns.histplot([getattr(i.data, param) for i in tp_ts], label="True Positive")
        sns.histplot([getattr(i.data, param) for i in fp_ts], label="False Positive")
        sns.histplot([getattr(i.data, param) for i in tn_ts], label="True Negative")
        sns.histplot([getattr(i.data, param) for i in fn_ts], label="False Negative")
        plt.title(param)
        plt.savefig(f"{out_prefix}/{domain}-{param}.pdf")

    hist("population_size")
    hist("migration_rate")
    hist("divergence_time")

if __name__ == "__main__":
    fire.Fire(plot)