from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Conv1D, AveragePooling1D, Dropout, Flatten, Dense  

from src.data.kerasSecondaryContactDataset import Dataset


source = Dataset("secondaryContact1/secondaryContact1-5000.json", 500, transpose=True)
validation = Dataset("secondaryContact1/secondaryContact1-val-500.json", 500, transpose=True)

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
model.add(Dense(2, activation="sigmoid"))

model.compile(loss='categorical_crossentropy', optimizer='adam', 
        metrics=["accuracy"])

model.fit(source.snps, source.migrationStates, 
        validation_data=(validation.snps, validation.migrationStates), 
        batch_size=64, epochs=10)

model.save("secondaryContact1/keras_conv1d_model")
