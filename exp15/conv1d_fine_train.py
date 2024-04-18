import sys, os
sys.path.append(os.path.abspath(".."))

from tensorflow import keras
from tensorflow import Variable
from keras.optimizers.legacy import Adam
from adapt.parameter_based import FineTuning
from adapt.utils import UpdateLambda
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSoftmax import predict
from conv1d_models import getEncoder, getTask
from src.kerasPlot import plotEncoded, plotFineTrainingAcc, plotFineTrainingLoss

prefix = "out/1d_fine"

snps = 3000

train = Dataset("../secondaryContact3/secondaryContact3-1000.json", snps, transpose=True, categorical=True)
validation = Dataset("../ghost3/ghost3-100.json", snps, transpose=True, categorical=True)

model = FineTuning(
    encoder=getEncoder(shape=train.shape),
    task=getTask(),
    optimizer=Adam(0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"])
history = model.fit(train.snps, train.migrationStates, epochs=50, batch_size=32, 
        validation_data=(validation.snps, validation.migrationStates)) 
model.save(f"{prefix}.keras")

test = Dataset("../secondaryContact3/secondaryContact3-test-100.json", snps, transpose=True, categorical=True)
ghost = Dataset("../ghost3/ghost3-test-100.json", snps, transpose=True, categorical=True)

plotFineTrainingAcc(model, f"{prefix}_training_acc.png")
plotFineTrainingLoss(model, f"{prefix}_training_loss.png")

np.savetxt(f"{prefix}_cm.txt", predict(model, test), fmt="%1.0f")
np.savetxt(f"{prefix}_cm-mispec.txt", predict(model, ghost), fmt="%1.0f")

plotEncoded(model, source=train, target=ghost, outputpath=f"{prefix}_encoded_tSNE.png")