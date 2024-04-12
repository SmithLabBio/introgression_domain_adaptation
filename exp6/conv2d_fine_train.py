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
from src.kerasPlot import plotEncoded

prefix = "out/secondaryContact1_conv2d_fine_tune"

snps = 1500

source = Dataset("../secondaryContact1/secondaryContact1-1000.json", snps, sorting=euclidean, split=True)

model = FineTuning(
    encoder=getEncoder(shape=source.shape),
    task=getTask(),
    optimizer=Adam(0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"])
history = model.fit(source.snps, source.migrationStates, epochs=50, batch_size=64)
model.save(f"{prefix}.keras")

test = Dataset("../secondaryContact1/secondaryContact1-test-100.json", snps, sorting=euclidean, split=True)
ghost = Dataset("../ghost1/ghost1-test-100.json", snps, sorting=euclidean, split=True)


np.savetxt(f"{prefix}_test_cm.txt", predict(model, test), fmt="%1.0f")
np.savetxt(f"{prefix}_ghost_cm.txt", predict(model, ghost), fmt="%1.0f")

plotEncoded(model, source=source, target=ghost, outputpath=f"{prefix}_encoded_tSNE.png")