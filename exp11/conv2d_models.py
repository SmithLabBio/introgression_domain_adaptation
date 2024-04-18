from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense  

def getEncoder(shape):
    model = Sequential()
    model.add(Input(shape=shape))
    model.add(Conv2D(10, [4,2], activation="relu"))
    model.add(MaxPool2D([2,2]))
    model.add(Flatten())
    model.add(Dense(20, activation="relu"))
    return model

def getTask():
    model = Sequential()
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation="softmax"))
    return model

def getDiscriminator():
    model = Sequential()
    model.add(Dense(100, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model