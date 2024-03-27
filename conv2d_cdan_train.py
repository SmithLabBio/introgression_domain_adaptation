from tensorflow import keras
from keras import Sequential, models
from keras.layers import Input, Conv1D, AveragePooling1D, Dropout, Flatten, Dense  
from keras.optimizers.legacy import Adam
from adapt.feature_based import CDAN

from src.data.kerasSecondaryContactDataset import Dataset
from kerasPredictSigmoid import predict

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
    return model

def getTask():
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="sigmoid"))
    return model

def getDiscriminator():
    model = Sequential()
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model

source = Dataset("secondaryContact1/secondaryContact1-5000.json", 500, transpose=True)
target = Dataset("ghost1/ghost1-5000.json", 500, transpose=True)

model = CDAN(
    lambda_=1,
    encoder=getEncoder(shape=source.shape), 
    task=getTask(), 
    discriminator=getDiscriminator(),
    optimizer=Adam(0.001), 
    loss="categorical_crossentropy",
    metrics=["accuracy"])
history = model.fit(source.snps, source.migrationStates, target.snps, 
                    epochs=20, batch_size=64)
# model.save("ghost1/conv1d_cdan_model")

test = Dataset("ghost1/ghost1-test-500.json", 500, transpose=True)
print(model.score(target.snps, target.migrationStates))
print(predict(model, test))