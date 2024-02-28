import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # silence tensorflow message 

import pickle
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Conv1D, Conv2D, AveragePooling1D, AveragePooling2D, Dropout, Flatten, Dense  
from keras.models import load_model
from keras.utils import to_categorical
from contextlib import redirect_stdout
import tskit
from data import getData

inDir = "secondaryContact1"

nSnps = 500
outDir = f"{inDir}/twoChannel"

data, snpMatrices, migrationStates = getData(
        f"{inDir}/secondaryContact1-1000.pickle", nSnps, split=True)

snpMatrices = np.transpose(snpMatrices, (0,2,3,1)) # Make last dimension the channel dimension
migrationStates = to_categorical(migrationStates)

nSamples = data["nSamples"] * 2 # Multipy by ploidy 

model = Sequential()
model.add(Input(shape=(nSamples, nSnps, 2)))
model.add(Conv2D(32, kernel_size=2, activation='relu'))
model.add(Conv2D(32, kernel_size=2, activation='relu'))
model.add(AveragePooling2D(pool_size=4))
model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=2, activation="relu"))
model.add(AveragePooling2D(pool_size=4))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation="sigmoid"))

model.compile(loss='categorical_crossentropy', optimizer='adam')
with open(f"{outDir}/model-summary.txt", "w") as fh:
    with redirect_stdout(fh):
        model.summary()
model.fit(snpMatrices, migrationStates, batch_size=32, epochs=5)
model.save(f"{outDir}/weights.keras")


