import torch
from torch import nn
from torch.optim import Adam
from lightning import LightningModule
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.functional import accuracy


class Lightning(LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model 
        self.save_hyperparameters()
        self.confusionMatrix = ConfusionMatrix(task="multiclass", num_classes=2)

    def forward(self, snps, distances):
        return self.model(snps, distances)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=0.001)

    def training_step(self, batch, batchIdx):
        snps, distances, migrationState = batch
        yhat = self(snps, distances)
        loss = nn.functional.cross_entropy(yhat, migrationState.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def evaluate(self, batch, stage=None):
        snps, distances, migrationState = batch
        yhat = self(snps, distances)
        loss = nn.functional.cross_entropy(yhat, migrationState.view(-1))
        preds = torch.argmax(yhat, dim=1)
        acc = accuracy(preds, migrationState, task="binary")
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_accuracy", acc, prog_bar=True)
        return preds, migrationState
        
    def validation_step(self, batch, batchIdx):
        self.evaluate(batch, "validation")

    def test_step(self, batch, batchIdx):
        pred, migState = self.evaluate(batch, "test")
        self.confusionMatrix(pred, migState)