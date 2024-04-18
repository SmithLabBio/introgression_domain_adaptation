import sys, os
sys.path.append(os.path.abspath(".."))

from tensorflow import keras
from keras.optimizers.legacy import Adam
from scipy.spatial.distance import euclidean
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSoftmax import predict
from conv2d_models import getEncoder, getTask
from src.kerasPlot import plotTrainingAcc, plotTrainingLoss

prefix = "out/2d_ghost"

snps = 1500

source = Dataset("../ghost3/ghost3-1000.json", snps, sorting=euclidean, split=True, categorical=True)
validation = Dataset("../ghost3/ghost3-100.json", snps, sorting=euclidean, split=True, categorical=True)

model = getEncoder(shape=source.shape)
for i in getTask().layers:
    model.add(i)

model.compile(loss='categorical_crossentropy', metrics=["accuracy"], 
        optimizer=Adam(0.0001)) 
history = model.fit(source.snps, source.migrationStates, 
        validation_data=(validation.snps, validation.migrationStates), 
        batch_size=32, epochs=50)
model.save(f"{prefix}.keras")

test = Dataset("../ghost3/ghost3-test-100.json", snps, sorting=euclidean, split=True, categorical=True)

plotTrainingAcc(history, f"{prefix}_training_acc.png")
plotTrainingLoss(history, f"{prefix}_training_loss.png")

np.savetxt(f"{prefix}_cm.txt", predict(model, test), fmt="%1.0f")

