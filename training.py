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


def run(dataPath, outPath=""):
    if not outPath:
        outPath = f"{path.splitext(dataPath)[0]}.pt"
    else:
        if not path.splitext(outPath)[1] == ".pt":
            quit(f"Aborted: outPath should have extension \".pt\"")
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

    trainingLossList = []
    # trainingAccuracy = [] # TODO: Compute accuracy
    validationLossList = []
    # validationAccuracy = []

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        # Training
        cumulativeTrainingLoss = 0.0 
        for x, y in trainLoader:
            model.train()
            yhat = model(x)
            loss = lossFunction(yhat, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() # Don't run between `loss.backward()` and `optimizer.step()` but anywhere else is fine. Tutorials online vary.
            cumulativeTrainingLoss += loss # TODO: Do I need to account for last batch size being smaller? 
        trainingLoss = cumulativeTrainingLoss / len(trainLoader)  
        trainingLossList.append(trainingLossList)

        # Validation
        cumulativeValidationLoss = 0.0
        with torch.no_grad():
            for x, y in validationLoader:
                yhat = model(x)
                loss = lossFunction(yhat, y.unsqueeze(1))
                cumulativeValidationLoss += loss
        validationLoss = cumulativeValidationLoss / len(validationLoader)
        validationLossList.append(validationLoss)

        print(f"Epoch {epoch + 1}: training loss = {trainingLoss} -- validation loss = {validationLoss}")

    trainingLoss = dict(trainingLoss=trainingLossList, validationLoss=validationLossList)
    torch.save([modelParams, model.state_dict(), trainingLoss], outPath) # Save current state, there is also option to save the best performing model state


if __name__ == "__main__":
    fire.Fire(run)