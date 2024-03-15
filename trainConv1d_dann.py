from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Conv1D, Conv2D, AveragePooling1D, AveragePooling2D, Dropout, Flatten, Dense  
from keras.utils import to_categorical
from keras.optimizers import Adam
from adapt.feature_based import DANN
from adapt.parameter_based import FineTuning

import numpy as np
from src.data.simulation import Simulations
from src.data.secondaryContact import SecondaryContactConfig, SecondaryContactData

from tskit import TreeSequence, Variant


def getGenotypeMatrix(ts: TreeSequence, nSnps: int, transpose=False) -> np.ndarray:
    var = Variant(ts, samples=ts.samples()) 
    if transpose:
        shape = (nSnps, len(ts.samples()))
    else:
        shape = (len(ts.samples()), nSnps)
    mat = np.empty(shape=shape)
    for site in range(nSnps):
        var.decode(site)
        if transpose:
            mat[site, :] = var.genotypes
        else:
            mat[:, site] = var.genotypes
    return mat

class Dataset():
    def __init__(self, path: str, nSnps: int, transpose=False):
        with open(path, "rb") as fh:
            jsonData = fh.read()
        self.simulations = Simulations[SecondaryContactConfig].model_validate_json(jsonData)
        snpMatrices = []
        migrationStates = []
        for ix, s in enumerate(self.simulations):
            ts = s.treeSequence 
            snpMatrices.append(getGenotypeMatrix(ts, nSnps, transpose=transpose))
            migrationStates.append(self.simulations[ix].data["migrationState"])
        self.snps = np.array(snpMatrices)
        # self.migrationStates = np.array(migrationStates)
        self.migrationStates = to_categorical(migrationStates, num_classes=2)
        self.shape = self.snps.shape[1:]

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
    model.compile(optimizer=Adam(0.01), loss="categorical_crossentropy")
    return model

def getTask():
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="sigmoid"))
    model.compile(optimizer=Adam(0.01), loss="categorical_crossentropy")
    return model

def getDiscriminator():
    model = Sequential()
    model.add(Dense(2048, activation="relu"))
    model.add(Dense(2048, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.01), loss="categorical_crossentropy")
    return model

source = Dataset("secondaryContact1/secondaryContact1-100-val.json", 400, transpose=True)
target = Dataset("secondaryContact1/secondaryContact1-100-val.json", 400, transpose=True)

model = DANN(encoder=getEncoder(shape=source.shape), task=getTask(), 
             discriminator=getDiscriminator(), Xt=target.snps, lambda_=1.0, 
            optimizer=Adam(0.001), loss="categorical_crossentropy",
            metrics=["accuracy"], copy=False)

# model = FineTuning(encoder=getEncoder(shape=source.shape), task=getTask(), discriminator=getDiscriminator(), lambda_=1.0, 
#         optimizer=Adam(0.001), loss="categorical_crossentropy",
#         metrics=["accuracy"], copy=False, pretrain=False)

model.fit(source.snps, source.migrationStates, epochs=1)
# model.save("ghost/conv1d_dann.keras")







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
