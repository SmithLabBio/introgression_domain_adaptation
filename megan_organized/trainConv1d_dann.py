from tensorflow import Variable
from keras.optimizers import Adam
from adapt.feature_based import DANN
import numpy as np
from adapt.utils import UpdateLambda

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredict import predict
from src.kerasPlot import plotEncoded, plotTrainingAcc, plotTrainingLoss
from src.models import getEncoder, getTask, getDiscriminator

# read data
source = Dataset("secondaryContact1/secondaryContact1-train.json", 1500, transpose=False, multichannel=True)
test = Dataset("secondaryContact1/secondaryContact1-test.json", 1500, transpose=False, multichannel=True)
val = Dataset("secondaryContact1/secondaryContact1-val.json", 1500, transpose=False, multichannel=True)
ghost = Dataset("ghost1/ghost1-test.json", 1500, transpose=False, multichannel=True)



# define and train dann model
lambda_max = 10
epochs = 10
batch_size = 32
learning_rate = 0.0001


dann = DANN(
    lambda_=Variable(0.0),
    encoder=getEncoder(shape=source.shape), 
    task=getTask(), 
    discriminator=getDiscriminator(),
    optimizer = Adam(learning_rate),
    optimizer_enc = Adam(learning_rate),
    optimizer_disc = Adam(learning_rate*50),  
    loss="categorical_crossentropy",
    copy = False,
    metrics=["accuracy"],
    callbacks=[UpdateLambda(lambda_max=lambda_max)])
history = dann.fit(source.snps, source.migrationStates, ghost.snps, 
                    epochs=epochs, batch_size=batch_size)
dann.save_weights("ghost1/conv1d_dann_v1a_model.model")

# plt training
plotTrainingAcc(dann, "results/training_acc_dann.png")
plotTrainingLoss(dann, "results/training_loss_dann.png")

# make predictions
np.savetxt("results/dann_test_cm.txt", predict(dann, test))
np.savetxt("results/dann_ghost_cm.txt", predict(dann, ghost))

# plot encoded space
plotEncoded(dann, source=source, target=ghost, outputpath="results/encoded_tSNE_dann.png")
