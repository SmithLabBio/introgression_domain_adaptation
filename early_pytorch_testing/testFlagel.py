#!/usr/bin/env python

import pandas as pd
from data import Dataset
import torch
from torch.utils.data import DataLoader
from flagel import Flagel


nSnps = 500
dataset = Dataset(path="scenario-1/scenario-1-test.npz", nSnps=nSnps)
loader = DataLoader(dataset, batch_size=64, shuffle=False)
config, state, _ = torch.load("scenario-1-flagel.pt")
model = Flagel(dataset.config["nSamples"], nSnps) 
model.load_state_dict(state)
predicted = []
target = []
with torch.inference_mode():
    for x, _, y in loader:
        yhat = model(x)
        predicted.extend(yhat.squeeze(1).tolist())
        target.extend(y.tolist())
df = pd.DataFrame.from_dict(dict(predicted=predicted, target=target))
df.to_csv("scenario-1-flagel.csv", index=False)