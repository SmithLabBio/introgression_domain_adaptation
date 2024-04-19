from keras.optimizers import Adam
from adapt.parameter_based import FineTuning
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredict import predict
from src.kerasPlot import plotEncoded
from src.models_v2 import getEncoder, getTask



# read data
source = Dataset("secondaryContact3/secondaryContact3-train.json", 1500, transpose=False, multichannel=True)
test = Dataset("secondaryContact3/secondaryContact3-test.json", 1500, transpose=False, multichannel=True)
val = Dataset("secondaryContact3/secondaryContact3-val.json", 1500, transpose=False, multichannel=True)
ghost = Dataset("ghost3/ghost3-test.json", 1500, transpose=False, multichannel=True)

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

# make predictions with original network for test data and ghost data
np.savetxt("original_test_cm.txt", predict(finetunig, test), fmt="%1.0f")
np.savetxt("original_ghost_cm.txt", predict(finetunig, ghost), fmt="%1.0f")


## plot encoded space for original
plotEncoded(finetunig, source=source, target=ghost, outputpath="encoded_tSNE_original.png")
