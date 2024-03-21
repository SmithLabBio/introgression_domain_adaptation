from tensorflow import keras
from keras import Sequential, models
from keras.layers import Input, Conv1D, AveragePooling1D, Dropout, Flatten, Dense  
from keras.optimizers.legacy import Adam
from adapt.feature_based import CDAN

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredict import predict

def getEncoder(shape):
    model = Sequential()
    model.add(Input(shape=shape))
    model.add(Conv1D(256, kernel_size=2, activation='relu'))
    model.add(Conv1D(128, kernel_size=2, activation='relu'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(128, kernel_size=2, activation="relu"))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    # model.compile(optimizer=Adam(0.01), loss="categorical_crossentropy")
    return model

def getTask():
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="sigmoid"))
    # model.compile(optimizer=Adam(0.01), loss="categorical_crossentropy")
    return model

def getDiscriminator():
    model = Sequential()
    model.add(Dense(2048, activation="relu"))
    model.add(Dense(2048, activation="relu"))
    model.add(Dense(1))
    # model.compile(optimizer=Adam(0.01), loss="categorical_crossentropy")
    return model

source = Dataset("secondaryContact1/secondaryContact1-5000.json", 500, transpose=True)
target = Dataset("ghost1/ghost1-5000.json", 500, transpose=True)

model = CDAN(
    encoder=getEncoder(shape=source.shape), 
    task=getTask(), 
    optimizer=Adam(0.001), 
    loss="categorical_crossentropy",
    metrics=["accuracy"])
history = model.fit(source.snps, source.migrationStates, target.snps, 
                    epochs=20, batch_size=64)
model.save("ghost1/conv1d_cdan_model")








# from torch.utils.data import DataLoader
# from lightning import Trainer

# from data.dataset import Dataset 
# from src.models.conv1d import Model
# # from src.lightning.lightningClassify import Lightning
# from src.models.conv1d_dann import Generator, Classifier, getDiscriminator

# import pytorch_lightning as pl
# import torch

# from pytorch_adapt.adapters import DANN
# from pytorch_adapt.containers import Models, Optimizers
# from pytorch_adapt.datasets import (DataloaderCreator, get_mnist_mnistm, 
#     CombinedSourceAndTargetDataset, SourceDataset, TargetDataset)
# from pytorch_adapt.frameworks.lightning import Lightning
# from pytorch_adapt.frameworks.utils import filter_datasets
# from pytorch_adapt.models import getDiscriminator, mnistC, mnistG
# from pytorch_adapt.validators import IMValidator

# # from src.models.conv1d_dann import Lightning

# outDir = "out/conv1d-1/"

# src_train = Dataset("secondaryContact1/secondaryContact1-1000.json", 400, split=False)
# src_val = Dataset("secondaryContact1/secondaryContact1-100-val.json", 400, split=False)
# target_train = Dataset("secondaryContact1/secondaryContact1-100-target-train.json", 400, split=False)
# target_val = Dataset("secondaryContact1/secondaryContact1-100-target-val.json", 400, split=False)
# train = CombinedSourceAndTargetDataset(SourceDataset(src_train), TargetDataset(target_train)) 

# datasets = dict(
#     src_train = src_train,
#     src_val = src_val,
#     target_train = target_train,
#     target_val = target_val,
#     train = train)

# dc = DataloaderCreator(train_kwargs=dict(batch_size=64, shuffle=True), val_kwargs=dict(batch_size=64))
# dataloaders = dc(**datasets)


# G = Generator(nSamples=100)
# C = Classifier() 
# D = getDiscriminator()

# # G_opt = torch.optim.Adam(G.parameters())
# # C_opt = torch.optim.Adam(C.parameters())
# # D_opt = torch.optim.Adam(D.parameters())

# models = Models({"G": G, "C": C, "D": D})
# optimizers = Optimizers((torch.optim.Adam, {"lr": 0.0001}))

# adapter = DANN(models=models, optimizers=optimizers)
# validator = IMValidator()
# dataloaders = dc(**filter_datasets(datasets, validator))
# train_loader = dataloaders.pop("train")

# L_adapter = Lightning(adapter, validator=validator)
# trainer = pl.Trainer(max_epochs=2)

# trainer.fit(L_adapter, train_loader, list(dataloaders.values()))
