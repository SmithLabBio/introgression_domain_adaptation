#!/usr/bin/env python 

import fire
import json
import pandas as pd
from os import listdir, sep 
from os.path import join, normpath, split
import os
import glob

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

def summarize(glob_pattern, directory, outpath, stat="mean", format="pprint"):
    dfs = []
    for p in glob.glob(glob_pattern):
        name = os.path.basename(p)
        data = []
        for dir in listdir(p): 
            with open(join(p, dir, directory, "stats.json")) as fh:
                data.append(json.load(fh))
            # try:
                # with open(join(p, dir, directory, "stats.json")) as fh:
                    # data.append(json.load(fh))
            # except:
            #     pass
        match stat:
            case "mean":
                df = pd.DataFrame(data).mean().to_frame().T
            case "max":
                df = pd.DataFrame(data).max().to_frame().T
        df.insert(0, "name", name)
        dfs.append(df)
    conc_df = pd.concat(dfs, axis=0)
    match format:
        case "pprint":
            with open(outpath, "w") as fh:
                print(conc_df, file=fh)
        case "csv":
            conc_df.to_csv(outpath, index=False)

if __name__ == "__main__":
    fire.Fire(summarize)
