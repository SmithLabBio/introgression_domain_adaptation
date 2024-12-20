#!/usr/bin/env python

import os
import pandas as pd
import fire

def summarize(path):
    means = []
    names = []
    for param in os.listdir(path): 
        entries = []
        for pop in os.listdir(f"{path}/{param}"):
            for rep  in os.listdir(f"{path}/{param}/{pop}"):
                if os.path.exists(f"{path}/{param}/{pop}/{rep}/history.csv"):
                    df = pd.read_csv(f"{path}/{param}/{pop}/{rep}/history.csv")
                    entry = df.iloc[-1].to_dict()
                    if "disc_acc" in df.columns:
                        entry["max_disc_acc"] = df["disc_acc"].max()
                    entries.append(entry)
        means.append(pd.DataFrame(entries).mean())
        names.append(param)

    df = pd.DataFrame(means)
    df.index = names # Ignore pycharm
    df = df.sort_index()
    df.to_csv(f"{path}/training-summary.csv")

if __name__ == "__main__":
    fire.Fire(summarize)