import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataTheta import Dataset
# from netTheta import Flagel
from netTheta import model
from tqdm import tqdm

epochs = 10 
nSnps = 500
dataset = Dataset(path="theta-sims-1000.pickle", nSnps=nSnps)

trainLoader = DataLoader(dataset, batch_size=64, shuffle=True) 

lossFunc = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in range(1, epochs+1):
    runningLoss = 0.0
    for x, y in tqdm(trainLoader):
        optimizer.zero_grad()
        yhat = model(x)
        loss = lossFunc(yhat, y)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
    print(f"Epoch {i}: loss = {runningLoss / len(trainLoader):.2e}")
torch.save(model.state_dict(), "weights.pt")


# epochs = 1 
# trainLossList = []
# valLossList = []


# for epoch in range(epochs):
#     cumTrainLoss = 0.0 
#     for x, y in trainLoader:
#         optimizer.zero_grad()
#         yhat = model(x)
#         loss = lossFunction(yhat, y.unsqueeze(1))
#         loss.backward()
#         optimizer.step()
#         cumTrainLoss += loss 
#     trainLoss = cumTrainLoss / len(trainLoader)  
#     trainLossList.append(trainLoss)

#     # Validation
#     cumValLoss = 0.0
#     with torch.no_grad():
#         for x, y in validationLoader:
#             yhat = model(x)
#             loss = lossFunction(yhat, y.unsqueeze(1))
#             cumValLoss += loss
#     valLoss = cumValLoss / len(validationLoader)
#     valLossList.append(valLoss)

#     print(f"Epoch {epoch + 1}: training loss = {trainLoss} -- validation loss = {valLoss}")

# trainingLoss = dict(trainingLoss=trainLossList, validationLoss=valLossList)
# torch.save(model.state_dict(), "weights.pt")
