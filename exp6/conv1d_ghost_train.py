import sys, os
sys.path.append(os.path.abspath(".."))

from tensorflow import keras
from keras.optimizers.legacy import Adam
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSigmoid import predict
from conv1d_models import getEncoder, getTask
from src.kerasPlot import plotTrainingAcc, plotTrainingLoss

prefix = "out/ghost1_conv1d"

snps = 1500

source = Dataset("../ghost1/ghost1-1000.json", snps, transpose=True)
validation = Dataset("../ghost1/ghost1-100.json", snps, transpose=True)

model = getEncoder(shape=source.shape)
for i in getTask().layers:
    model.add(i)

model.compile(loss='binary_crossentropy', metrics=["accuracy"], 
        optimizer=Adam(0.0001)) 
model.fit(source.snps, source.migrationStates, 
        validation_data=(validation.snps, validation.migrationStates), 
        batch_size=64, epochs=50)

test = Dataset("../ghost1/ghost1-test-100.json", snps, transpose=True)

model.save(f"{prefix}.keras")

plotTrainingAcc(model, f"{prefix}_training_acc.png")
plotTrainingLoss(model, f"{prefix}_training_loss.png")

np.savetxt(f"{prefix}_test_cm.txt", predict(model, test), fmt="%1.0f")


