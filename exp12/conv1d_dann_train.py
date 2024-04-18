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
from src.kerasPredictSoftmax import predict
from conv1d_models import getEncoder, getTask, getDiscriminator
from src.kerasPlot import plotEncoded, plotAdaptTrainingAcc, plotAdaptTrainingLoss

prefix = "out/1d_dann"

snps = 3000

source = Dataset("../secondaryContact3/secondaryContact3-1000.json", snps, transpose=True, categorical=True)
target = Dataset("../ghost3/ghost3-100.json", snps, transpose=True, categorical=True)

model = DANN(
    lambda_=1,
    encoder=getEncoder(shape=source.shape), 
    task=getTask(), 
    discriminator=getDiscriminator(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    optimizer=Adam(0.00001))
history = model.fit(source.snps, source.migrationStates, target.snps, epochs=50, batch_size=32)
model.save(f"{prefix}.keras")

test = Dataset("../secondaryContact3/secondaryContact3-test-100.json", snps, transpose=True, categorical=True)
ghost = Dataset("../ghost3/ghost3-test-100.json", snps, transpose=True, categorical=True)

plotAdaptTrainingAcc(model, f"{prefix}_training_acc.png")
plotAdaptTrainingLoss(model, f"{prefix}_training_loss.png")

np.savetxt(f"{prefix}_cm.txt", predict(model, test), fmt="%1.0f")
np.savetxt(f"{prefix}_cm-mispec.txt", predict(model, ghost), fmt="%1.0f")

plotEncoded(model, source=source, target=ghost, outputpath=f"{prefix}_encoded_tSNE.png")