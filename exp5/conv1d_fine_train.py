import sys, os
sys.path.append(os.path.abspath(".."))

from tensorflow import keras
from tensorflow import Variable
from keras.optimizers.legacy import Adam
from adapt.parameter_based import FineTuning
from adapt.utils import UpdateLambda
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSigmoid import predict
from conv1d_models import getEncoder, getTask
from src.kerasPlot import plotEncoded

prefix = "out/secondaryContact2_conv1d_fine_tune"

snps = 1500

source = Dataset("../secondaryContact2/secondaryContact2-1000.json", snps, transpose=True)

model = FineTuning(
    encoder=getEncoder(shape=source.shape),
    task=getTask(),
    optimizer=Adam(0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"])
history = model.fit(source.snps, source.migrationStates, epochs=50, batch_size=64)
model.save(f"{prefix}.keras")

test = Dataset("../secondaryContact2/secondaryContact2-test-100.json", snps, transpose=True)
ghost = Dataset("../ghost2/ghost2-test-100.json", snps, transpose=True)


np.savetxt(f"{prefix}_test_cm.txt", predict(model, test), fmt="%1.0f")
np.savetxt(f"{prefix}_ghost_cm.txt", predict(model, ghost), fmt="%1.0f")

plotEncoded(model, source=source, target=ghost, outputpath=f"{prefix}_encoded_tSNE.png")