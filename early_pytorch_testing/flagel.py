#!/usr/bin/env python3

from torch import nn
# from torch.nn.functional import relu, sigmoid

class Flagel(nn.Module):
    def __init__(self, samplesPerPop, nSnps, ploidy=2):
        super().__init__()
        nHaplotypes = samplesPerPop * 2 * ploidy
        self.net = nn.Sequential(
            nn.Conv1d(nHaplotypes, 256, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv1d(128, 128, kernel_size=2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        return out 