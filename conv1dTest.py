from torch.utils.data import DataLoader
from lightning import Trainer

from data import Dataset 
from conv1dModel import CNN 



testDataset = Dataset("../secondaryContact1/secondaryContact1-100.json", 400, split=False)
testLoader = DataLoader(testDataset, batch_size=32)

nSamples = testDataset.simulations.config.nSamples 
model = CNN.load_from_checkpoint("conv1d.ckpt", nSamples=nSamples * 4)
trainer = Trainer()
trainer.test(model, testLoader)
print(model.confusionMatrix.compute())

# fig, ax = model.confmat.plot()
# fig.show()

# classes = ["no migration", "migration"]
# cm = confusion_matrix(testDataset.migrationStates, predicted, labels=[0,1])
# rep = classification_report(testDataset.migrationStates, predicted, target_names=classes, 
#         labels=[0,1])

# df = pd.DataFrame(cm, index=classes, columns=classes)
# # df.to_csv(f"{outDir}/confusionMatrix.csv")
# # with open(f"{outDir}/classification-report.txt", "w") as fh:
#     # fh.write(str(rep))
# print("\n*********************************************************************")
# print("Confusion Matrix\n")
# print(df)
# print("\n*********************************************************************")
# print("Classification Report\n")
# print(rep)

# sns.heatmap(df, annot=True, cmap="Blues")
# plt.savefig(f"{outDir}/confusion-matrix.png")





# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torch.optim import Adam
# from lightning import LightningModule, Trainer
# from data import Dataset
# from sklearn.metrics import confusion_matrix, classification_report
# import pandas as pd
# import seaborn as sns

# class CNN(LightningModule):
#     def __init__(self, nSamples: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(nSamples, 256, 2), nn.ReLU(),
#             nn.Conv1d(256, 128, 2), nn.ReLU(),
#             nn.AvgPool1d(2),
#             nn.Dropout(0.25),
#             nn.Conv1d(128, 128, 2), nn.ReLU(),
#             nn.AvgPool1d(2),
#             nn.Dropout(0.25),
#             nn.Flatten(),
#             nn.LazyLinear(128), nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 128), nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 2),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.net(x)

#     def configure_optimizers(self):
#         return Adam(self.net.parameters(), lr=0.001)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         yhat = self(x)
#         loss = nn.functional.cross_entropy(yhat, y.view(-1))
#         return loss

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         yhat = self(x)
#         loss = nn.functional.cross_entropy(yhat, y.view(-1))
#         pred = torch.argmax(yhat, dim=1)
#         accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)
#         output = dict(loss=loss, accuracy=torch.tensor(accuracy), 
#                 probabilities=yhat, predicted=pred)
#         return pred 
    
#     # def predict_step(self, batch, batch_idx):
#     #     x, y = batch
#     #     pred = self.net(x)
#     #     return pred



# testDataset = Dataset("secondaryContact1/secondaryContact1-100.pickle", 500)
# testLoader = DataLoader(testDataset, batch_size=64)

# model = CNN.load_from_checkpoint("conv1dPT.ckpt", nSamples=testDataset.nSamples * 4)
# trainer = Trainer()
# t = trainer.test(model=model, dataloaders=testLoader)
# print(t)

# # classes = ["no migration", "migration"]
# # cm = confusion_matrix(testDataset.migrationStates, predicted, labels=[0,1])
# # rep = classification_report(testDataset.migrationStates, predicted, target_names=classes, 
# #         labels=[0,1])

# # df = pd.DataFrame(cm, index=classes, columns=classes)
# # # df.to_csv(f"{outDir}/confusionMatrix.csv")
# # # with open(f"{outDir}/classification-report.txt", "w") as fh:
# #     # fh.write(str(rep))
# # print("\n*********************************************************************")
# # print("Confusion Matrix\n")
# # print(df)
# # print("\n*********************************************************************")
# # print("Classification Report\n")
# # print(rep)

# # sns.heatmap(df, annot=True, cmap="Blues")
# # plt.savefig(f"{outDir}/confusion-matrix.png")

