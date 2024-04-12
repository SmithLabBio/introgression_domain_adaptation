import sys, os
sys.path.append(os.path.abspath(".."))

from tensorflow import keras
from tensorflow import Variable
from keras.optimizers.legacy import Adam
from adapt.parameter_based import FineTuning
from adapt.utils import UpdateLambda
from scipy.spatial.distance import euclidean
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSigmoid import predict
from conv2d_models import getEncoder, getTask
from src.kerasPlot import plotEncoded, plotTrainingAcc, plotTrainingLoss

prefix = "out/secondaryContact2_conv2d_fine_tune"

snps = 500

source = Dataset("../secondaryContact2/secondaryContact2-5000.json", snps, sorting=euclidean, split=True)

model = FineTuning(
    encoder=getEncoder(shape=source.shape),
    task=getTask(),
    optimizer=Adam(0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"])
history = model.fit(source.snps, source.migrationStates, epochs=50, batch_size=64)
model.save(f"{prefix}.keras")

test = Dataset("../secondaryContact2/secondaryContact2-test-500.json", snps, sorting=euclidean, split=True)
ghost = Dataset("../ghost2/ghost2-test-500.json", snps, sorting=euclidean, split=True)

plotTrainingAcc(model, f"{prefix}_training_acc.png")
plotTrainingLoss(model, f"{prefix}_training_loss.png")

np.savetxt(f"{prefix}_test_cm.txt", predict(model, test), fmt="%1.0f")
np.savetxt(f"{prefix}_ghost_cm.txt", predict(model, ghost), fmt="%1.0f")

plotEncoded(model, source=source, target=ghost, outputpath=f"{prefix}_encoded_tSNE.png")