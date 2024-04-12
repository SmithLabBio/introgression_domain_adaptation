import sys, os
sys.path.append(os.path.abspath(".."))

from tensorflow import keras
from keras.optimizers.legacy import Adam
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSigmoid import predict
from conv2d_models import getEncoder, getTask
from src.kerasPlot import plotTrainingAcc, plotTrainingLoss

prefix = "out/secondaryContact2_conv1d"

snps = 500

source = Dataset("../secondaryContact2/secondaryContact2-5000.json", snps, transpose=True)
validation = Dataset("../secondaryContact2/secondaryContact2-val-500.json", snps, transpose=True)

model = getEncoder(shape=source.shape)
for i in getTask().layers:
    model.add(i)

model.compile(loss='binary_crossentropy', metrics=["accuracy"], 
        optimizer=Adam(0.0001)) 
model.fit(source.snps, source.migrationStates, 
        validation_data=(validation.snps, validation.migrationStates), 
        batch_size=64, epochs=50)

test = Dataset("../secondaryContact2/secondaryContact2-test-500.json", snps, transpose=True)
ghost = Dataset("../ghost2/ghost2-test-500.json", snps, transpose=True)

model.save(f"{prefix}.keras")

plotTrainingAcc(model, f"{prefix}_training_acc.png")
plotTrainingLoss(model, f"{prefix}_training_loss.png")

np.savetxt(f"{prefix}_test_cm.txt", predict(model, test), fmt="%1.0f")
np.savetxt(f"{prefix}_ghost_cm.txt", predict(model, ghost), fmt="%1.0f")


