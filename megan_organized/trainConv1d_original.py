from keras.optimizers import Adam
from adapt.parameter_based import FineTuning
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredict import predict
from src.kerasPlot import plotEncoded
from src.models import getEncoder, getTask



# read data
source = Dataset("secondaryContact1/secondaryContact1-train.json", 1500, transpose=False, multichannel=True)
test = Dataset("secondaryContact1/secondaryContact1-test.json", 1500, transpose=False, multichannel=True)
val = Dataset("secondaryContact1/secondaryContact1-val.json", 1500, transpose=False, multichannel=True)
ghost = Dataset("ghost1/ghost1-test.json", 1500, transpose=False, multichannel=True)

# train the original model
learning_rate = 0.0001
epochs = 10
batch_size = 32

finetunig = FineTuning(encoder=getEncoder(shape=source.shape),
                         task=getTask(),
                         optimizer=Adam(learning_rate),
                         optimizer_enc=Adam(learning_rate),
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])
history_finetunig = finetunig.fit(source.snps, source.migrationStates, epochs = epochs, batch_size = batch_size, validation_data=(val.snps, val.migrationStates))
finetunig.save_weights("ghost1/conv1d_finetunig_v1a_model.model")

# make predictions with original network for test data and ghost data
np.savetxt("results/original_test_cm.txt", predict(finetunig, test))
np.savetxt("results/original_ghost_cm.txt", predict(finetunig, ghost))


## plot encoded space for original
plotEncoded(finetunig, source=source, target=ghost, outputpath="results/encoded_tSNE_original.png")
