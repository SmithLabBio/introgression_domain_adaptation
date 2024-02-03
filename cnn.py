#!/usr/bin/env python3

import numpy as np
import pickle
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import relu, sigmoid
import fire
# from torchinfo import summary
# from torchviz import make_dot
import pandas as pd
import yaml
from pydantic import BaseModel

class Block(nn.Module):
    def __init__(self, iteration: int, nOutputs: int, nFeatures: int, nFullyConnected: int):
        super().__init__() 
        self.nOutputs = nOutputs
        self.nFullyConnected = nFullyConnected
        if iteration == 0: 
            featureLayers = 2
        else:
            featureLayers = 3
        self.conv = nn.Conv2d(in_channels=nFeatures * featureLayers, 
                out_channels=nFeatures, kernel_size=(1, 3))
        self.bn = nn.BatchNorm2d(num_features=nFeatures * featureLayers)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2))
        self.fc = nn.Linear(in_features=3 * nFullyConnected, out_features=nOutputs)

    def forward(self, x, output):
        half = x.size(2) // 2
        x = self.conv(self.bn(x))
        colMeans0 = torch.mean(x, 2, keepdim=True)
        colMeans1 = torch.mean(x[:, :, 0:half , :], 2, keepdim=True) 
        colMeans2 = torch.mean(x[:, :, half:-1, :], 2, keepdim=True) 
        totMean0 = torch.mean(colMeans0[:, :self.nFullyConnected, :, :], 3).squeeze(2)
        totMean1 = torch.mean(colMeans1[:, :self.nFullyConnected, :, :], 3).squeeze(2)
        totMean2 = torch.mean(colMeans2[:, :self.nFullyConnected, :, :], 3).squeeze(2)
        catMeans = torch.cat((totMean0, totMean1, totMean2), 1) 
        output += self.fc(catMeans)
        colMeansExp0 = colMeans0.expand(-1, -1, x.size(2), -1)
        colMeansExp1 = colMeans1.expand(-1, -1, half, -1)
        colMeansExp2 = colMeans2.expand(-1, -1, half, -1)
        colMeansExpCat = torch.cat((colMeansExp1, colMeansExp2), 2)
        x = torch.cat((x, colMeansExp0, colMeansExpCat), 1) 
        x = relu(self.maxpool(x))
        return x, output


class CNN(nn.Module):
    def __init__(self, nBlocks: int, nFeatures: int, nOutputs: int, nFullyConnected:int):
        """
        nFullyConnected: number of filters from from each filter group fed to fully connected layer
        """
        super().__init__()
        assert nFeatures >= nFullyConnected
        self.nFeatures = nFeatures
        self.nOutputs = nOutputs
        self.nFullyConnected = nFullyConnected
        self.posConv = nn.Conv2d(in_channels=1, out_channels=nFeatures, kernel_size=(1, 3))
        self.posConvBn = nn.BatchNorm2d(num_features=nFeatures)
        self.snpConv = nn.Conv2d(in_channels=1, out_channels=nFeatures, kernel_size=(1, 3))
        self.snpConvBn = nn.BatchNorm2d(num_features=nFeatures)
        self.blocks = nn.ModuleList(
                [Block(i, nOutputs, nFeatures, nFullyConnected) for i in range(nBlocks)])

    def forward(self, x):
        snp = x[:, 1:, :].unsqueeze(1) 
        snp = relu(self.snpConvBn(self.snpConv(snp)))
        pos = x[:, 0, :].view(x.shape[0], 1, 1, -1) 
        pos = relu(self.posConvBn(self.posConv(pos)))
        pos = pos.expand(-1, -1, snp.size(2), -1)
        x = torch.cat((pos, snp), 1) 
        output = torch.zeros(size=(x.size(0), self.nOutputs), device=x.device)
        for block in self.blocks:
            x, output = block(x, output)
        return output



## Show details of model
# testResult = model(torch.randint(0, 4, (2, 51, 400)).float())
# g = make_dot(result.mean(), params=dict(model.named_parameters()))
# g.view("cnn")
# # print(summary(model, input_size=(1, 51, 400))) # Batch size, Rows, Cols. Channel dimension is added in forward function


# def run():

# if __name__ == "__main__":
#     fire.Fire(run)


