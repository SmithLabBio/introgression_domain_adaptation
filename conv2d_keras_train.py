from tensorflow import keras
from keras import Sequential
from keras.layers import concatenate
from keras.optimizers.legacy import Adam
from scipy.spatial.distance import euclidean

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSigmoid import predict
from conv2d_models import getEncoder, getTask

snps = 1500

source = Dataset("secondaryContact1/secondaryContact1-1000.json", snps, 
        sorting=euclidean, split=True)
validation = Dataset("secondaryContact1/secondaryContact1-val-100.json", snps, 
        sorting=euclidean, split=True)

model = getEncoder(shape=source.shape)
for i in getTask().layers:
    model.add(i)

model.compile(loss='binary_crossentropy', metrics=["accuracy"], 
        optimizer=Adam(0.0001)) 
model.fit(source.snps, source.migrationStates, 
        validation_data=(validation.snps, validation.migrationStates), 
        batch_size=64, epochs=10)

model.save("out/conv2d_secondaryContact1.keras")

test = Dataset("secondaryContact1/secondaryContact1-test-100.json", snps,
        sorting=euclidean, split=True)
print("Specified model")
print(predict(model, test))

test = Dataset("ghost1/ghost1-test-100.json", snps,
        sorting=euclidean, split=True)
print("Mispecified model")
print(predict(model, test))