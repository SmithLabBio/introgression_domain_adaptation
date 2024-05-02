#!/usr/bin/env python

import fire
from src.data.simulation import Simulations
import matplotlib.pyplot as plt

def plot(path):
    with open(path, "rb") as fh:
        data = fh.read()
    sims = Simulations.model_validate_json(data)
    fst = []
    migrationRate = []
    for i in sims:
        ts = i.treeSequence
        fst.append(ts.Fst((ts.samples(3), ts.samples(4))))
        migrationRate.append(i.data["migrationRate"])
    plt.scatter(migrationRate, fst)
    plt.xlabel("Migration Rate")
    plt.ylabel("Fst")
    # plt.hist(fst, bins=10, color='blue', edgecolor='black')
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Data')
    plt.savefig("fst.png")

if __name__ == "__main__":
    fire.Fire(plot)