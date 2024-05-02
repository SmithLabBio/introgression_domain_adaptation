from torch import nn

# model = nn.Sequential(
#     nn.Conv2d(1, 32, (1,2)), nn.ReLU(),
#     nn.Conv2d(32, 32, (1,2)), nn.ReLU(),
#     nn.AvgPool2d((1,2)),
#     nn.Dropout(0.25),
#     nn.Conv2d(32, 32, (1,2)), nn.ReLU(),
#     nn.AvgPool2d((1,2)),
#     nn.Dropout(0.25),
#     nn.Flatten(),
#     nn.LazyLinear(32), nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(32, 32), nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(32, 2)
# )



# from torchinfo import summary
# print(summary(model, input_size=(1, 1, 100, 500)))