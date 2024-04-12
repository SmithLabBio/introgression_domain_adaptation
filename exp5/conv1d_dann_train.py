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
from conv1d_models import getEncoder, getTask, getDiscriminator
from src.kerasPlot import plotEncoded, plotAdaptTrainingAcc, plotAdaptTrainingLoss

prefix = "out/secondaryContact2_conv1d_dann"

snps = 500

source = Dataset("../secondaryContact2/secondaryContact2-1000.json", snps, transpose=True)
target = Dataset("../ghost2/ghost2-100.json", snps, transpose=True)

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

test = Dataset("../secondaryContact2/secondaryContact2-test-100.json", snps, transpose=True)
ghost = Dataset("../ghost2/ghost2-test-100.json", snps, transpose=True)

plotAdaptTrainingAcc(model, f"{prefix}_training_acc.png")
plotAdaptTrainingLoss(model, f"{prefix}_training_loss.png")

np.savetxt(f"{prefix}_test_cm.txt", predict(model, test), fmt="%1.0f")
np.savetxt(f"{prefix}_ghost_cm.txt", predict(model, ghost), fmt="%1.0f")

plotEncoded(model, source=source, target=ghost, outputpath=f"{prefix}_encoded_tSNE.png")