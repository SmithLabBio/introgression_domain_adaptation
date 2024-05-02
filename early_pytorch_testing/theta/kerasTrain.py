import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # silence tensorflow message 

import pickle
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv1D, AveragePooling1D, Dropout, Flatten, Dense  
from keras.models import load_model


def loadData(path, nSnps):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    popSizes = data["popSizes"]
    fullSnpMatrices = data["charMatrices"]
    snpMatrices = []
    for i in fullSnpMatrices:
        snpMatrices.append(i[:, :nSnps].transpose())
    snpMatrices = np.array(snpMatrices)
    return snpMatrices, popSizes

snpMatrices, popSizes = loadData("theta-sims-1000.pickle", 500)

model = Sequential()
# model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(100, 500)))
model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(500, 100)))
model.add(Conv1D(64, kernel_size=2, activation='relu'))
model.add(AveragePooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(snpMatrices, popSizes, batch_size=32, epochs=10)
model.save("keras-theta-weights.keras")


# Testing 
import pandas as pd

model = load_model("keras-theta-weights.keras")
testSnpMatrices, testPopSizes = loadData("theta-sims-100.pickle", 500)
predicted = model.predict(testSnpMatrices).squeeze(1)
df = pd.DataFrame.from_dict(dict(predicted=predicted, target=testPopSizes))
df.to_csv("keras-theta-predicted.csv", index=False)

# Plotting 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("keras-theta-predicted.csv")
plt.scatter(df["target"], df["predicted"])
plt.axline((0,0), slope=1)
plt.title("Theta")
plt.xlabel("Target")
plt.ylabel("Predicted")
plt.savefig("keras-theta-predicted.png")
plt.clf()

