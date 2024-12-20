#!/usr/bin/env python

import fire
import matplotlib.pyplot as plt
import pandas as pd
import os

# def run(inpath, outpath):
def run(dir, outpath):
    for i in os.listdir(dir): 
        inpath = os.path.join(dir, i, "history.csv")
        df = pd.read_csv(inpath)
        plt.axhline(y=0.5, color="grey", linestyle="dashed")
        # plt.plot(df["accuracy"], color="darkgreen", label=f"Accuracy: {df['accuracy'].iloc[-1]:0.2f}")
        # plt.plot(df["disc_acc"], color="forestgreen", linestyle="dashed", label=f"Discrimination Accuracy: {df['disc_acc'].iloc[-1]:0.2f}")
        # plt.plot(df["val_accuracy"], color="green", linestyle="dotted", label=f"Validation Accuracy: {df['val_accuracy'].iloc[-1]:0.2f}")
        plt.plot(df["accuracy"], label=f"Accuracy: {df['accuracy'].iloc[-1]:0.2f}")
        plt.plot(df["disc_acc"], label=f"Discrimination Accuracy: {df['disc_acc'].iloc[-1]:0.2f}")
        plt.plot(df["val_accuracy"], label=f"Validation Accuracy: {df['val_accuracy'].iloc[-1]:0.2f}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        # plt.title("Training History")
        plt.legend(loc="lower right", bbox_to_anchor=(1.0, -0.35))
        # plt.savefig(outpath, bbox_inches="tight")
        plt.savefig(os.path.join(dir, i, outpath), bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    fire.Fire(run)

