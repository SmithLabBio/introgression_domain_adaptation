from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Conv1D, AveragePooling1D, Dropout, Flatten, Dense  
from keras.optimizers.legacy import Adam

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSigmoid import predict


source = Dataset("secondaryContact1/secondaryContact1-1000.json", 500, transpose=True)
validation = Dataset("secondaryContact1/secondaryContact1-val-100.json", 500, transpose=True)

model = Sequential()
model.add(Input(shape=source.shape))
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
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', metrics=["accuracy"], 
        optimizer=Adam(0.0001)) 

model.fit(source.snps, source.migrationStates, 
        validation_data=(validation.snps, validation.migrationStates), 
        batch_size=64, epochs=40)

model.save("out/conv1d_secondaryContact1.keras")

test = Dataset("secondaryContact1/secondaryContact1-test-100.json", 500, transpose=True)
print("Specified model")
print(predict(model, test))

test = Dataset("ghost1/ghost1-test-100.json", 500, transpose=True)
print("Mispecified model")
print(predict(model, test))