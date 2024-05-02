import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # silence tensorflow message 

import pickle
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Conv1D, Conv2D, AveragePooling1D, AveragePooling2D, Dropout, Flatten, Dense  
from keras.utils import to_categorical
from contextlib import redirect_stdout
import tskit
from data import getData

inDir = "secondaryContact1" 

################################################################################
# 1D Convolution

# Snp dimension first
transpose = True

# Haplotype dimensions first
# transpose = False 

nSnps = 500
data, snpMatrices, migrationStates = getData(
        f"{inDir}/secondaryContact1-1000.pickle", nSnps, transpose=transpose)
migrationStates = to_categorical(migrationStates)
nSamples = data["nSamples"] * 2 * 2 # Multiply by ploidy and by population number 

if transpose:
    shape = (nSnps, nSamples)
    outDir = f"{inDir}/singleChannel-1d-transposed"
else:
    shape = (nSamples, nSnps)
    outDir = f"{inDir}/singleChannel-1d"

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
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation="sigmoid"))

model.compile(loss='categorical_crossentropy', optimizer='adam')

with open(f"{outDir}/model-summary.txt", "w") as fh:
    with redirect_stdout(fh):
        model.summary()

model.fit(snpMatrices, migrationStates, batch_size=32, epochs=10)
model.save(f"{outDir}/weights.keras")



# ################################################################################
# # 2D Convolution
# nSnps = 500
# outDir = f"{inDir}/singleChannel-2d"
# data, snpMatrices, migrationStates = getData(
#         f"{inDir}/secondaryContact1-1000.pickle", nSnps)
# migrationStates = to_categorical(migrationStates)
# nSamples = data["nSamples"] * 2 * 2 # Multiply by ploidy and by population number 

# snpMatrices = np.expand_dims(snpMatrices, axis=3)

# model = Sequential()
# model.add(Input(shape=(nSamples, nSnps, 1)))
# model.add(Conv2D(64, kernel_size=2, activation='relu'))
# model.add(Conv2D(64, kernel_size=2, activation='relu'))
# model.add(AveragePooling2D(pool_size=2))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, kernel_size=2, activation="relu"))
# model.add(AveragePooling2D(pool_size=2))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation="sigmoid"))

# model.compile(loss='categorical_crossentropy', optimizer='adam')
# with open(f"{outDir}/model-summary.txt", "w") as fh:
#     with redirect_stdout(fh):
#         model.summary()
# model.fit(snpMatrices, migrationStates, batch_size=32, epochs=5)
# model.save(f"{outDir}/weights.keras")
