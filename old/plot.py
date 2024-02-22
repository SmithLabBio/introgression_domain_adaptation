#!/usr/bin/env python

import pandas as pd
from matplotlib import pyplot as plt
import fire
import os.path as path

def run(inPath, outPrefix="", force=False):
    if not outPrefix:
        outPath = f"{path.splitext(inPath)[0]}.png"
    else:
        outPath = f"{outPrefix}.png"
    if path.exists(outPath): 
        if not force:
            quit(f"Aborted: {outPath} already exists")
    df = pd.read_csv(inPath)
    plt.scatter(df["target"], df["predicted"])
    plt.title("Prediction")
    plt.xlabel("Target")
    plt.ylabel("Predicted")
    plt.savefig(outPath)
    plt.clf()

# df = pd.read_csv("scenario-1-loss.csv")
# plt.plot(df["trainingLoss"])
# plt.plot(df["validationLoss"])
# plt.title("Training MSE")
# plt.xlabel("Epoch")
# plt.ylabel("MSE Loss")
# plt.savefig("scenario-1-loss.png")

if __name__ == "__main__":
    fire.Fire(run)