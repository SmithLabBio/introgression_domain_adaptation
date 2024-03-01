import torch
from torch import nn
from torch.optim import Adam
from lightning import LightningModule
from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchmetrics.functional import accuracy
from torch.nn.functional import relu


class SPIDNABlock(nn.Module):
    def __init__(self, n_outputs, n_features):
        super().__init__()
        self.n_outputs = n_outputs
        self.phi = nn.Conv2d(n_features * 2, n_features, (1, 3))
        self.phi_bn = nn.BatchNorm2d(n_features * 2)
        self.maxpool = nn.MaxPool2d((1, 2))
        self.fc = nn.Linear(n_outputs, n_outputs)

    def forward(self, x, output):
        x = self.phi(self.phi_bn(x))
        psi1 = torch.mean(x, 2, keepdim=True)
        psi = psi1
        current_output = self.fc(torch.mean(psi[:, :self.n_outputs, :, :], 3).squeeze(2))
        output += current_output
        psi = psi.expand(-1, -1, x.size(2), -1)
        x = torch.cat((x, psi), 1)
        x = relu(self.maxpool(x))
        return x, output


class SPIDNA(nn.Module):
    def __init__(self, n_blocks, n_features, n_outputs):
        super().__init__()
        self.n_outputs = n_outputs
        self.conv_pos = nn.Conv2d(1, n_features, (1, 3))
        self.conv_pos_bn = nn.BatchNorm2d(n_features)
        self.conv_snp = nn.Conv2d(1, n_features, (1, 3))
        self.conv_snp_bn = nn.BatchNorm2d(n_features)
        self.blocks = nn.ModuleList([SPIDNABlock(n_outputs, n_features)
                                     for i in range(n_blocks)])

    def forward(self, snp, pos):
        snp = snp.unsqueeze(1) # Add channel dimension
        pos = pos.view(pos.shape[0], 1, 1, -1) # Add channel and row dim
        pos = relu(self.conv_pos_bn(self.conv_pos(pos)))
        pos = pos.expand(-1, -1, snp.size(2), -1)
        snp = relu(self.conv_snp_bn(self.conv_snp(snp)))
        x = torch.cat((pos, snp), 1)
        output = torch.zeros(x.size(0), self.n_outputs, device=x.device)
        for block in self.blocks:
            x, output = block(x, output)
        return output


class CNN(LightningModule):
    def __init__(self, n_blocks: int, n_features: int, n_outputs: int):
        super().__init__()

        self.model = SPIDNA(n_blocks, n_features, n_outputs)
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