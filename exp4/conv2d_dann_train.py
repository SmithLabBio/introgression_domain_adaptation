import sys, os
sys.path.append(os.path.abspath(".."))

from tensorflow import keras
from tensorflow import Variable
from keras.optimizers.legacy import Adam
from adapt.feature_based import DANN
from adapt.utils import UpdateLambda
from scipy.spatial.distance import euclidean
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSigmoid import predict
from conv2d_models import getEncoder, getTask, getDiscriminator
from src.kerasPlot import plotEncoded, plotTrainingAcc, plotTrainingLoss

prefix = "out/secondaryContact1_conv2d_dann"

snps = 1500 

source = Dataset("../secondaryContact1/secondaryContact1-5000.json", snps, sorting=euclidean, split=True)
target = Dataset("../ghost1/ghost1-500.json", snps, sorting=euclidean, split=True)

model = DANN(
    lambda_=Variable(0.0),
    encoder=getEncoder(shape=source.shape), 
    task=getTask(), 
    discriminator=getDiscriminator(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
    optimizer=Adam(0.0001),
    callbacks=[UpdateLambda(lambda_max=10)]) 
history = model.fit(source.snps, source.migrationStates, target.snps, epochs=50, batch_size=64)
model.save(f"{prefix}.keras")

test = Dataset("../secondaryContact1/secondaryContact1-test-500.json", snps, sorting=euclidean, split=True)
ghost = Dataset("../ghost1/ghost1-test-500.json", snps, sorting=euclidean, split=True)

plotTrainingAcc(model, f"{prefix}_training_acc.png")
plotTrainingLoss(model, f"{prefix}_training_loss.png")

np.savetxt(f"{prefix}_test_cm.txt", predict(model, test), fmt="%1.0f")
np.savetxt(f"{prefix}_ghost_cm.txt", predict(model, ghost), fmt="%1.0f")

plotEncoded(model, source=source, target=ghost, outputpath=f"{prefix}_encoded_tSNE.png")