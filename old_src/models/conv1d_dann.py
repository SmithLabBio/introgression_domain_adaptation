import torch
from torch import nn
from torch.optim import Adam
from lightning import LightningModule
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.functional import accuracy
from ..data.dataset import Data


class Generator(nn.Module):
    def __init__(self, nSamples: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(nSamples, 256, 2), nn.ReLU(),
            nn.Conv1d(256, 128, 2), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.25),
            nn.Conv1d(128, 128, 2), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Identity() # Returns features? Not totally clear why this is needed
        )

    def forward(self, x: Data):
        return self.model(x.snps)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1))

    def forward(self, x):
        return self.net(x)


class Lightning(LightningModule):
    def __init__(self, nSamples: int):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.G = Generator(nSamples) 
        self.C = Classifier()
        self.D = Discriminator()
        # self.automatic_optimization = False
        self.confusionMatrix = ConfusionMatrix(task="multiclass", num_classes=2)

    # def forward(self, snps, distances):
        # return self.model(snps, distances)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=0.001)

    def training_step(self, batch, batchIdx):
        snps, distances, migrationState = batch
        # yhat = self(snps, distances)
        # loss = nn.functional.cross_entropy(yhat, migrationState.view(-1))
        # self.log("train_loss", loss, prog_bar=True)
        # return loss
    
    # def evaluate(self, batch, stage=None):
    #     snps, distances, migrationState = batch
    #     yhat = self(snps, distances)
    #     loss = nn.functional.cross_entropy(yhat, migrationState.view(-1))
    #     preds = torch.argmax(yhat, dim=1)
    #     acc = accuracy(preds, migrationState, task="binary")
    #     if stage:
    #         self.log(f"{stage}_loss", loss, prog_bar=True)
    #         self.log(f"{stage}_accuracy", acc, prog_bar=True)
    #     return preds, migrationState
        
    # def validation_step(self, batch, batchIdx):
    #     self.evaluate(batch, "validation")

    # def test_step(self, batch, batchIdx):
    #     pred, migState = self.evaluate(batch, "test")
    #     self.confusionMatrix(pred, migState)

# snps = torch.rand(32, 100, 400)
# dist = torch.rand(32, 400)

# features = Features(100)
# classifier = Classifier() 
# discriminator = Discriminator()

# f = features(snps, dist)
# c = classifier(f)
# d = discriminator(f)

# print(features)
# print(pred)
# print(clas)
