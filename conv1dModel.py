import torch
from torch import nn
from torch.optim import Adam
from lightning import LightningModule
from torchmetrics.classification import Accuracy, BinaryConfusionMatrix

class CNN(LightningModule):
    def __init__(self, nSamples: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(nSamples, 256, 2), nn.ReLU(),
            nn.Conv1d(256, 128, 2), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.25),
            nn.Conv1d(128, 128, 2), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.LazyLinear(128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        self.confusionMatrix = BinaryConfusionMatrix()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return Adam(self.net.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        snps, distances, migrationState = batch
        yhat = self(snps)
        loss = nn.functional.cross_entropy(yhat, migrationState.view(-1))
        return loss

    def test_step(self, batch, batch_idx):
        snps, distances, migrationState = batch
        yhat = self(snps)
        loss = nn.functional.cross_entropy(yhat, migrationState.view(-1))
        pred = torch.argmax(yhat, dim=1)
        # accuracy = torch.sum(migrationState == pred).item() / (len(y) * 1.0)
        self.confusionMatrix.update(pred, migrationState)
        self.log_dict(dict(loss=loss))#, accuracy=accuracy))
    
    # def predict_step(self, x, batch_idx):
    #     pred = self.net(x)
    #     return pred
