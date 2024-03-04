import torch
from torch import nn
from torch.optim import Adam
from lightning import LightningModule
from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchmetrics.functional import accuracy
from torch.nn.functional import relu


class Block(nn.Module):
    def __init__(self, nOutputs, nFeatures, nSamples):
        super().__init__()
        self.nOutputs = nOutputs
        self.nSamples = nSamples
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(nFeatures * 2),
            nn.Conv2d(nFeatures * 2, nFeatures, (1, 3)))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(nFeatures * 3),
            nn.Conv2d(nFeatures * 3, nFeatures, (1, 3)))
        self.maxpool = nn.MaxPool2d((1, 2))
        self.fc = nn.Linear(nFeatures, nOutputs)

    def forward(self, x, output, ix):
        if ix == 0:
            x = self.conv1(x)
        else:
            x = self.conv2(x)
        mean1 = torch.mean(x[:,:, 0:self.nSamples, :], 2, keepdim=True) 
        mean2 = torch.mean(x[:,:, self.nSamples:-1, :], 2, keepdim=True) 
        rowMean = torch.mean(x, 2, keepdim=True) 
        colMean = torch.mean(rowMean, 3).squeeze(2)
        currentOutput = self.fc(colMean)
        output += currentOutput
        mean1 = mean1.expand(-1, -1, x.size(2), -1)
        mean2 = mean2.expand(-1, -1, x.size(2), -1)
        x = torch.cat((x, mean1, mean2), 1)
        x = relu(self.maxpool(x))
        return x, output


class Model(nn.Module):
    def __init__(self, nBlocks, nFeatures, nOutputs, nSamples):
        super().__init__()
        self.nOutputs = nOutputs
        self.conv = nn.Sequential(
            nn.Conv2d(1, nFeatures, (1,3)),
            nn.BatchNorm2d(nFeatures),
            nn.ReLU())
        self.blocks = nn.ModuleList([Block(nOutputs, nFeatures, nSamples)
                                     for i in range(nBlocks)])

    def forward(self, snp, pos):
        snp = snp.unsqueeze(1) # Add channel dimension
        pos = pos.view(pos.shape[0], 1, 1, -1) # Add channel and row dim
        pos = self.conv(pos) 
        pos = pos.expand(-1, -1, snp.size(2), -1)
        snp = self.conv(snp)
        x = torch.cat((pos, snp), 1)
        output = torch.zeros(x.size(0), self.nOutputs, device=x.device)
        for ix, block in enumerate(self.blocks):
            x, output = block(x, output, ix)
        return output


# x = torch.rand(32, 100, 400)
# d = torch.rand(32, 400) 
# m = Model(5, 50, 2, 50)
# o = m(x, d)