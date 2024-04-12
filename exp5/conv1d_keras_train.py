import sys, os
sys.path.append(os.path.abspath(".."))

from tensorflow import keras
from keras.optimizers.legacy import Adam
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSigmoid import predict
from conv1d_models import getEncoder, getTask
from src.kerasPlot import plotTrainingAcc, plotTrainingLoss

prefix = "out/secondaryContact2_conv1d"

snps = 1500

source = Dataset("../secondaryContact2/secondaryContact2-1000.json", snps, transpose=True)
validation = Dataset("../secondaryContact2/secondaryContact2-val-100.json", snps, transpose=True)

model = getEncoder(shape=source.shape)
for i in getTask().layers:
    model.add(i)

model.compile(loss='binary_crossentropy', metrics=["accuracy"], 
        optimizer=Adam(0.0001)) 
history = model.fit(source.snps, source.migrationStates, 
        validation_data=(validation.snps, validation.migrationStates), 
        batch_size=64, epochs=50)
model.save(f"{prefix}.keras")

test = Dataset("../secondaryContact2/secondaryContact2-test-100.json", snps, transpose=True)
ghost = Dataset("../ghost2/ghost2-test-100.json", snps, transpose=True)

np.savetxt(f"{prefix}_test_cm.txt", predict(model, test), fmt="%1.0f")
np.savetxt(f"{prefix}_ghost_cm.txt", predict(model, ghost), fmt="%1.0f")

plotTrainingAcc(history, f"{prefix}_training_acc.png")
plotTrainingLoss(history, f"{prefix}_training_loss.png")
