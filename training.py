import cnn
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split


class Trainer():
    def __init__(self, model, lossFunction, optimizer):
        self.model = model
        self.lossFunction = lossFunction
        self.optimizer = optimizer
    
    def trainingStep(self, x, y):
        model.train()
        yhat = self.model(x)
        loss = self.lossFunction(yhat, y.unsqueeze(1))
        self.optimizer.zero_grad() # Don't run between `loss.backward()` and `optimizer.step()` but anywhere else is fine.
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def validationStep(self, x, y):
        yhat = self.model(x)
        loss = self.lossFunction(yhat, y.unsqueeze(1))
        return loss.item()


# prefix = "scenario-1"
prefix = "config"
batchSize = 64 
epochs = 10 
dataset = cnn.Dataset(path=f"{prefix}.npz", nSnps=500)
nTest = int(len(dataset) * 0.15)
nVal = int(len(dataset) * 0.15)
nTrain = len(dataset) - nTest - nVal 
trainSet, testSet, valSet = random_split(dataset, [nTrain, nTest, nVal])
trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True) 
testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=True) 
validationLoader = DataLoader(valSet, batch_size=batchSize, shuffle=True) 

model = cnn.CNN(nBlocks=3, nFeatures=10, nOutputs=1, nFullyConnected=10)
lossFunction = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
# trainer = Trainer(model, lossFunction, optimizer)

## Show details of model
# testResult = model(torch.randint(0, 4, (2, 51, 400)).float())
# g = make_dot(result.mean(), params=dict(model.named_parameters()))
# g.view("cnn")
# # print(summary(model, input_size=(1, 51, 400))) # Batch size, Rows, Cols. Channel dimension is added in forward function




trainingLossList = []
# trainingAccuracy = [] # TODO: Compute accuracy
validationLossList = []
# validationAccuracy = []

for epoch in range(epochs):
    cumulativeTrainingLoss = 0.0 
    for x, y in trainLoader:
        model.train()
        yhat = model(x)
        loss = lossFunction(yhat, y.unsqueeze(1))
        optimizer.zero_grad() # Don't run between `loss.backward()` and `optimizer.step()` but anywhere else is fine.
        loss.backward()
        optimizer.step()
        cumulativeTrainingLoss += loss # TODO: Do I need to account for last batch size being smaller? 

    trainingLoss = cumulativeTrainingLoss / len(trainLoader)  
    trainingLossList.append(trainingLossList)

    cumulativeValidationLoss = 0.0
    with torch.no_grad():
        for x, y in validationLoader:
            yhat = model(x)
            loss = lossFunction(yhat, y.unsqueeze(1))
            cumulativeValidationLoss += loss
    validationLoss = cumulativeValidationLoss / len(validationLoader)
    validationLossList.append(validationLoss)

    print(f"Epoch {epoch + 1}: training loss = {trainingLoss} -- validation loss = {validationLoss}")


torch.save([model.config, model.state_dict()], f"{prefix}-model-state.pt") # Save current state, there is also option to save the best performing model state
# df = pd.DataFrame.from_dict(dict(trainingLoss=trainingLossList, validationLoss=validationLossList))
# df.to_csv("scenario-1-loss.csv", index=False)
