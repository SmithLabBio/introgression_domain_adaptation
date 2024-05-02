#!/usr/bin/env python

import fire
import cnn
import data
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import os.path as path


def run(dataPath, outPrefix=""):
    if not outPrefix:
        outPath = f"{path.splitext(dataPath)[0]}.pt"
    else:
        outPath = f"{outPrefix}.pt"
    if path.exists(outPath): 
        quit(f"Aborted: {outPath} already exists")

    batchSize = 64 
    epochs = 10 

    dataset = data.Dataset(path=dataPath, nSnps=500)
    nVal = int(len(dataset) * 0.2)
    nTrain = len(dataset) - nVal 
    trainSet, valSet = random_split(dataset, [nTrain, nVal])
    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True) 
    validationLoader = DataLoader(valSet, batch_size=batchSize, shuffle=True) 

    modelParams = dict(nBlocks=3, nFeatures=10, nOutputs=1, nFullyConnected=10)
    model = cnn.CNN(**modelParams)

    lossFunction = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    trainLossList = []
    valLossList = []

    for epoch in range(1, epochs + 1):
        cumTrainLoss = 0.0 
        for x, y in trainLoader:
            optimizer.zero_grad() # Don't run between `loss.backward()` and `optimizer.step()` but anywhere else is fine. Tutorials online vary.
            yhat = model(x)
            loss = lossFunction(yhat, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            cumTrainLoss += loss # TODO: Do I need to account for last batch size being smaller? 
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
    torch.save([modelParams, model.state_dict(), trainingLoss], outPrefix) # Save current state, there is also option to save the best performing model state


if __name__ == "__main__":
    fire.Fire(run)