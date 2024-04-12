import sys, os
sys.path.append(os.path.abspath(".."))

from tensorflow import keras
from keras.optimizers.legacy import Adam
from scipy.spatial.distance import euclidean
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSigmoid import predict
from conv2d_models import getEncoder, getTask
from src.kerasPlot import plotTrainingAcc, plotTrainingLoss

prefix = "out/secondaryContact1_conv2d"

snps = 1500

source = Dataset("../secondaryContact1/secondaryContact1-1000.json", snps, sorting=euclidean, split=True)
validation = Dataset("../secondaryContact1/secondaryContact1-val-100.json", snps, sorting=euclidean, split=True)

model = getEncoder(shape=source.shape)
for i in getTask().layers:
    model.add(i)

model.compile(loss='binary_crossentropy', metrics=["accuracy"], 
        optimizer=Adam(0.0001)) 
history = model.fit(source.snps, source.migrationStates, 
        validation_data=(validation.snps, validation.migrationStates), 
        batch_size=64, epochs=50)
model.save(f"{prefix}.keras")

test = Dataset("../secondaryContact1/secondaryContact1-test-100.json", snps, sorting=euclidean, split=True)
ghost = Dataset("../ghost1/ghost1-test-100.json", snps, sorting=euclidean, split=True)

np.savetxt(f"{prefix}_test_cm.txt", predict(model, test), fmt="%1.0f")
np.savetxt(f"{prefix}_ghost_cm.txt", predict(model, ghost), fmt="%1.0f")

plotTrainingAcc(history, f"{prefix}_training_acc.png")
plotTrainingLoss(history, f"{prefix}_training_loss.png")