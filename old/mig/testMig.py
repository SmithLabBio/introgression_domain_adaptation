import torch
from torch.utils.data import DataLoader
from dataMig import Dataset
import pandas as pd
from netMig import model 
from sklearn.metrics import confusion_matrix, classification_report


nSnps = 500
dataset = Dataset(path="mig-sims-100.pickle", nSnps=nSnps)
state = torch.load("weights.pt")
model.load_state_dict(state)

loader = DataLoader(dataset, batch_size=64, shuffle=False)
predicted = []
# target = []
with torch.inference_mode():
    for x, y in loader:
        yhat = model(x)
        predicted.extend(torch.max(yhat.data, 1)[1].tolist())
        # target.extend(y.tolist())


confusion = confusion_matrix(dataset.migrationStates, predicted, labels=[0,1])
rep = classification_report(dataset.migrationStates, predicted, 
                            target_names=["none", "migration"], labels=[0,1])
print(confusion)
print(rep)
# df = pd.DataFrame.from_dict(dict(predicted=predicted, target=target))
# df.to_csv("theta-predicted.csv", index=False)