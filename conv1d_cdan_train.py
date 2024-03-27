from tensorflow import keras
from tensorflow import Variable
from keras import Sequential, models
from keras.layers import Input, Conv1D, AveragePooling1D, Dropout, Flatten, Dense  
from keras.optimizers.legacy import Adam
from adapt.feature_based import CDAN
from adapt.utils import UpdateLambda

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSigmoid import predict

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
    model.add(Dense(1, activation="sigmoid"))
    return model

def getDiscriminator():
    model = Sequential()
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model

source = Dataset("secondaryContact1/secondaryContact1-1000.json", 500, transpose=True)
target = Dataset("ghost1/ghost1-1000.json", 500, transpose=True)

model = CDAN(
    lambda_=Variable(0.0),
    encoder=getEncoder(shape=source.shape), 
    task=getTask(), 
    discriminator=getDiscriminator(),
    optimizer=Adam(0.0001), 
    loss="binary_crossentropy",
    metrics=["accuracy"],
    callbacks=[UpdateLambda(lambda_max=0.1)])

history = model.fit(source.snps, source.migrationStates, target.snps, 
                    epochs=40, batch_size=64)
model.save("out/conv1d_cdan_secondaryContact1.keras")

test = Dataset("ghost1/ghost1-test-100.json", 500, transpose=True)
print(model.score(target.snps, target.migrationStates))
print(predict(model, test))