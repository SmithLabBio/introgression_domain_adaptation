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

prefix = "out/ghost2_conv2d"

snps = 500

source = Dataset("../ghost2/ghost2-5000.json", snps, sorting=euclidean, split=True)
validation = Dataset("../ghost2/ghost2-val-500.json", snps, sorting=euclidean, split=True)

model = getEncoder(shape=source.shape)
for i in getTask().layers:
    model.add(i)

model.compile(loss='binary_crossentropy', metrics=["accuracy"], 
        optimizer=Adam(0.0001)) 
model.fit(source.snps, source.migrationStates, 
        validation_data=(validation.snps, validation.migrationStates), 
        batch_size=64, epochs=50)
model.save(f"{prefix}.keras")

test = Dataset("../ghost2/ghost2-test-500.json", snps, sorting=euclidean, split=True)

plotTrainingAcc(model, f"{prefix}_training_acc.png")
plotTrainingLoss(model, f"{prefix}_training_loss.png")

np.savetxt(f"{prefix}_test_cm.txt", predict(model, test), fmt="%1.0f")

