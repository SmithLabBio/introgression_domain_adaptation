from tensorflow import Variable
from keras.optimizers import Adam
from adapt.feature_based import CDAN
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



# define and train cdan model
lambda_max = 10
epochs = 50
batch_size = 32
learning_rate = 0.0001


cdan = CDAN(
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

history = cdan.fit(source.snps, source.migrationStates, ghost.snps, 
                    epochs=epochs, batch_size=batch_size)

cdan.save_weights("ghost1/conv1d_cdan_v1a_model.model")

# plt training
plotTrainingAcc(cdan, "results/training_acc_cdan.png")
plotTrainingLoss(cdan, "results/training_loss_cdan.png")

# make predictions
np.savetxt("results/cdan_test_cm.txt", predict(cdan, test))
np.savetxt("results/cdan_ghost_cm.txt", predict(cdan, ghost))

# plot encoded space
plotEncoded(cdan, source=source, target=ghost, outputpath="results/encoded_tSNE_cdan.png")

