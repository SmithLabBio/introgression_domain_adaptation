import torch
from torch.utils.data import DataLoader
from dataTheta import Dataset
import pandas as pd
from netTheta import model 


nSnps = 500
dataset = Dataset(path="theta-sims-100.pickle", nSnps=nSnps)
state = torch.load("weights.pt")
model.load_state_dict(state)

loader = DataLoader(dataset, batch_size=64, shuffle=False)
predicted = []
target = []
with torch.inference_mode():
    for x, y in loader:
        yhat = model(x)
        target.extend(y.squeeze(1).tolist())
        predicted.extend(yhat.squeeze(1).tolist())
df = pd.DataFrame.from_dict(dict(predicted=predicted, target=target))
df.to_csv("theta-predicted.csv", index=False)