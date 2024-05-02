from flagel import Flagel
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import relu, sigmoid
import torch.optim as optim
from data import Dataset


nSnps = 500
dataset = Dataset(path="scenario-1/scenario-1.npz", nSnps=nSnps)
nVal = int(len(dataset) * 0.2)
nTrain = len(dataset) - nVal 
trainSet, valSet = random_split(dataset, [nTrain, nVal])
trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True) 
validationLoader = DataLoader(valSet, batch_size=64, shuffle=True) 

model = Flagel(dataset.config["nSamples"], nSnps) # multply by number of populations and ploidy
# from torchinfo import summary
# print(summary(model, input_size=(64, dataset.config["nSamples"] * 4, 500)))


lossFunction = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

epochs = 5 
trainLossList = []
valLossList = []

for epoch in range(epochs):
    cumTrainLoss = 0.0 
    for x, _, y in trainLoader:
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
        for x, _, y in validationLoader:
            yhat = model(x)
            loss = lossFunction(yhat, y.unsqueeze(1))
            cumValLoss += loss
    valLoss = cumValLoss / len(validationLoader)
    valLossList.append(valLoss)

    print(f"Epoch {epoch + 1}: training loss = {trainLoss} -- validation loss = {valLoss}")

trainingLoss = dict(trainingLoss=trainLossList, validationLoss=valLossList)
torch.save([dict(), model.state_dict(), trainingLoss], "scenario-1-flagel.pt") # Save current state, there is also option to save the best performing model state

