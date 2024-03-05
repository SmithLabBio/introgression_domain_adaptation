from torch import nn

class Model(nn.Module):
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

    def forward(self, snps, distances):
        return self.model(snps)