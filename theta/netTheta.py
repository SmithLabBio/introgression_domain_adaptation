from torch import nn

class Flagel(nn.Module):
    def __init__(self, nSamples):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(nSamples * 2, 64, kernel_size=2),
            nn.ReLU(),
            # nn.Conv1d(256, 128, kernel_size=2),
            # nn.ReLU(),
            # nn.AvgPool1d(kernel_size=2),
            # nn.Dropout(0.25),
            nn.Conv1d(64, 64, kernel_size=2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )

    def forward(self, x): 
        out = self.net(x)
        return(out)