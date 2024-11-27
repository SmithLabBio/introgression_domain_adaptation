#!/usr/bin/env python 

import fire
import json
import pandas as pd
from os import listdir, sep 
from os.path import join, normpath, split
import os
import glob

# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', None)

def summarize(directory, test_dir_name, stat="mean"):
    dfs = []
    for params in os.listdir(directory): 
        params_path = f"{directory}/{params}"
        if os.path.isdir(params_path):
            data = []
            for rep in os.listdir(params_path):
                with open(f"{params_path}/{rep}/{test_dir_name}/stats.json") as fh:
                    data.append(json.load(fh))
            df = pd.DataFrame(data)
            match stat:
                case "mean":
                    df = df.mean().to_frame().T
                case "max":
                    df = df.mean().to_frame().T
            df.insert(0, "name", params)
            dfs.append(df)
    conc_df = pd.concat(dfs, axis=0)
    conc_df.to_csv(f"{directory}/testing-summary-{stat}.csv", index=False)

if __name__ == "__main__":
    fire.Fire(summarize)
