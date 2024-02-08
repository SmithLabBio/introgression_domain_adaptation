import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataTheta import Dataset
from netTheta import Flagel


nSnps = 500
dataset = Dataset(path="theta-sims.pickle", nSnps=nSnps)
model = Flagel(dataset.nSamples)

nVal = int(len(dataset) * 0.2)
nTrain = len(dataset) - nVal 
trainSet, valSet = random_split(dataset, [nTrain, nVal])
trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True) 
validationLoader = DataLoader(valSet, batch_size=64, shuffle=True) 

lossFunction = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

epochs = 20 
trainLossList = []
valLossList = []

for epoch in range(epochs):
    cumTrainLoss = 0.0 
    for x, y in trainLoader:
        optimizer.zero_grad()
        yhat = model(x)
        loss = lossFunction(yhat, y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        cumTrainLoss += loss 
    trainLoss = cumTrainLoss / len(trainLoader)  
    trainLossList.append(trainLoss)

    # Validation
    cumValLoss = 0.0
    with torch.no_grad():
        for x, y in validationLoader:
            yhat = model(x)
            loss = lossFunction(yhat, y.unsqueeze(1))
            cumValLoss += loss
    valLoss = cumValLoss / len(validationLoader)
    valLossList.append(valLoss)

    print(f"Epoch {epoch + 1}: training loss = {trainLoss} -- validation loss = {valLoss}")

trainingLoss = dict(trainingLoss=trainLossList, validationLoss=valLossList)
torch.save(model.state_dict(), "weights.pt")
