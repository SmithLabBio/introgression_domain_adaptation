from torch import nn

model = nn.Sequential(
    nn.Conv2d(1, 256, (1,2)), nn.ReLU(),
    nn.Conv2d(256, 128, (1,2)), nn.ReLU(),
    nn.AvgPool2d((1,2)),
    nn.Dropout(0.25),
    nn.Conv2d(128, 128, (1,2)), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,2)),
    nn.Dropout(0.25),
    nn.Flatten(),
    nn.LazyLinear(128), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 2),
    nn.Sigmoid()
)

# from torchinfo import summary
# print(summary(model, input_size=(1, 1, 100, 500)))