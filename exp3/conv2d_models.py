from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense  

def getEncoder(shape):
    model = Sequential()
    model.add(Input(shape=shape))
    model.add(Conv2D(32, [3,3], activation="relu"))
    model.add(MaxPool2D([2,2]))
    model.add(Conv2D(32, [3,3], activation="relu"))
    model.add(MaxPool2D([2,2]))
    model.add(Conv2D(32, [3,3], activation="relu"))
    model.add(MaxPool2D([2,2]))
    model.add(Flatten())
    return model

def getTask():
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    return model

def getDiscriminator():
    model = Sequential()
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model