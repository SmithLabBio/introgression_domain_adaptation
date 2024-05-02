import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # silence tensorflow message 

import pickle
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv1D, AveragePooling1D, Dropout, Flatten, Dense  
from keras.models import load_model
from keras.utils import to_categorical


def loadData(path, nSnps):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    migrationStates = data["migrationStates"]
    fullSnpMatrices = data["charMatrices"]
    snpMatrices = []
    for i in fullSnpMatrices:
        snpMatrices.append(i[:, :nSnps].transpose())
    snpMatrices = np.array(snpMatrices)
    return snpMatrices, migrationStates

snpMatrices, migrationStates = loadData("mig-sims-1000.pickle", 500)
migrationStates = to_categorical(migrationStates)

model = Sequential()
model.add(Conv1D(256, kernel_size=2, activation='relu', input_shape=(500, 100)))
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
model.fit(snpMatrices, migrationStates, batch_size=32, epochs=10)
model.save("keras-mig-weights.keras")


