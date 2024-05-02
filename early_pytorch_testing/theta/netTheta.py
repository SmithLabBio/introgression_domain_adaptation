from torch import nn

model = nn.Sequential(
    nn.Conv2d(1, 64, (1,2)), nn.ReLU(),
    nn.Conv2d(64, 64, (1,2)), nn.ReLU(),
    nn.AvgPool2d((1,2)),
    nn.Dropout(0.25),
    nn.Flatten(),
    nn.LazyLinear(64), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 64),
    nn.Dropout(0.5),
    nn.Linear(64, 1),
)



# from torchinfo import summary
# print(summary(model, input_size=(1, 1, 100, 500)))