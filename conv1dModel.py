import torch
from torch import nn
from torch.optim import Adam
from lightning import LightningModule
from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchmetrics.functional import accuracy

class CNN(LightningModule):
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
            nn.LazyLinear(128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        self.confusionMatrix = ConfusionMatrix(task="multiclass", num_classes=2)
        # self.testAccuracy = Accuracy(task="multiclass", num_classes=2)
        # self.valAccuracy = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        snps, distances, migrationState = batch
        yhat = self(snps)
        loss = nn.functional.cross_entropy(yhat, migrationState.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def evaluate(self, batch, stage=None):
        snps, distances, migrationState = batch
        yhat = self(snps)
        loss = nn.functional.cross_entropy(yhat, migrationState.view(-1))
        preds = torch.argmax(yhat, dim=1)
        acc = accuracy(preds, migrationState, task="binary")
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_accuracy", acc, prog_bar=True)
        return preds, migrationState
        
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "validation")

    def test_step(self, batch, batch_idx):
        pred, migState = self.evaluate(batch, "test")
        # self.testAccuracy(pred, migrationState)
        self.confusionMatrix(pred, migState)
    
    # def predict_step(self, x, batch_idx):
    #     pred = self.model(x)
    #     return pred
