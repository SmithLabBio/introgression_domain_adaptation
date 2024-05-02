from torch import nn
from torch.optim import Adam
from lightning import LightningModule
from torchmetrics.classification import Accuracy, ConfusionMatrix

class Lightning(LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model 
        self.predicted = []
        self.target = []

    def forward(self, snps, distances):
        return self.model(snps, distances)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        snps, distances, theta = batch
        yhat = self(snps, distances)
        loss = nn.functional.mse_loss(yhat, theta)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def evaluate(self, batch, stage=None):
        snps, distances, theta = batch
        yhat = self(snps, distances)
        loss = nn.functional.mse_loss(yhat, theta)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
        return yhat, theta
        
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "validation")

    def test_step(self, batch, batch_idx):
        pred, target  = self.evaluate(batch, "test")
        self.predicted.extend(pred)
        self.target.extend(target)