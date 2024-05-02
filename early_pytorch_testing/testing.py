#!/usr/bin/env python

import pandas as pd
import fire
from cnn import CNN
from data import Dataset
import os.path as path
import torch
from torch.utils.data import DataLoader


def run(statePath, dataPath, outPrefix=""):
    if not outPrefix:
        outPath = f"{path.splitext(dataPath)[0]}-predicted.csv"
    else:
        outPath = f"{outPrefix}.csv"
    if path.exists(outPath): 
        quit(f"Aborted: {outPath} already exists")

    data = Dataset(path=dataPath, nSnps=500)
    loader = DataLoader(data, batch_size=64, shuffle=False)
    config, state, _ = torch.load(statePath)
    model = CNN(**config) 
    model.load_state_dict(state)
    model.eval()
    predicted = []
    target = []
    with torch.inference_mode():
        for x, y in loader:
            yhat = model(x)
            predicted.extend(yhat.squeeze(1).tolist())
            target.extend(y.tolist())
    df = pd.DataFrame.from_dict(dict(predicted=predicted, target=target))
    df.to_csv(outPath, index=False)

if __name__ == "__main__":
    fire.Fire(run)