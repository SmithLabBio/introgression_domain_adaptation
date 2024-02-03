#!/usr/bin/env python

import pandas as pd
import torch
import fire
import cnn


def run(statePath, dataPath):
    data = cnn.Dataset(path=dataPath, nSnps=500)
    loader = cnn.DataLoader(data, batch_size=64, shuffle=False)
    config, state = torch.load(statePath)
    model = cnn.CNN(**config) 
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
    df.to_csv("scenario-1-predicted.csv", index=False)


if __name__ == "__main__":
    fire.Fire(run)