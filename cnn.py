#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn.functional import relu
import fire
from torchinfo import summary


class Block(nn.Module):
    def __init__(self, nOutputs: int, nFeatures: int):
        super().__init__()
        self.nOutputs = nOutputs
        self.phi = nn.Conv2d(in_channels=nFeatures * 2, out_channels=nFeatures, kernel_size=(1, 3))
        self.phiBn = nn.BatchNorm2d(num_features=nFeatures * 2)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2))
        self.fc = nn.Linear(in_features=nOutputs, out_features=nOutputs)

    def forward(self, x, output):
        x = self.phi(self.phiBn(x))
        psi1 = torch.mean(x, 2, keepdim=True)
        psi = psi1
        current_output = self.fc(torch.mean(psi[:, :self.nOutputs, :, :], 3).squeeze(2))
        output += current_output
        psi = psi.expand(-1, -1, x.size(2), -1)
        x = torch.cat((x, psi), 1)
        x = relu(self.maxpool(x))
        return x, output


class CNN(nn.Module):
    def __init__(self, nBlocks: int, nFeatures: int, nOutputs: int):
        super().__init__()
        self.nOutputs = nOutputs
        self.posConv = nn.Conv2d(in_channels=1, out_channels=nFeatures, kernel_size=(1, 3))
        self.posConvBn = nn.BatchNorm2d(num_features=nFeatures)
        self.snpConv = nn.Conv2d(in_channels=1, out_channels=nFeatures, kernel_size=(1, 3))
        self.snpConvBn = nn.BatchNorm2d(num_features=nFeatures)
        self.blocks = nn.ModuleList([Block(nOutputs, nFeatures)
                                     for i in range(nBlocks)])

    def forward(self, x):
        snp = x[:, 1:, :].unsqueeze(1) # Unsqueeze adds another dimension to the 
        # tenosor slice at position 1, creating a channel dimension of length 1, 
        # TODO: why not use `view` here because unsqueeze creates a new copy. 
        snp = relu(self.snpConvBn(self.snpConv(snp)))

        pos = x[:, 0, :].view(x.shape[0], 1, 1, -1) # View reshapes the tensor 
        # slice to [batch size=input batch size, channel=1, height=1, width=original dimension] 
        pos = relu(self.posConvBn(self.posConv(pos)))
        pos = pos.expand(-1, -1, snp.size(2), -1)

        x = torch.cat((pos, snp), 1) # Concatenate along channel dimension 
        output = torch.zeros(x.size(0), self.nOutputs, device=x.device)
        for block in self.blocks:
            x, output = block(x, output)
        return output

model = CNN(7, 50, 1)
print(summary(model, input_size=(1, 51, 400))) # Batch size, Rows, Cols. Channel dimension is added in forward function

x = torch.randint(0, 4, (2, 51, 400)).float()
result = model(x)
print(model)
print(result)



# def run(dataPath):
#     data = np.load(dataPath, allow_pickle=True)
#     charMatrices = data["charMatrices"]
#     # tensor = torch.Tensor(charMatrices) # pytorch cant take array of numpy.objects

# if __name__ == "__main__":
#     fire.Fire(run)

