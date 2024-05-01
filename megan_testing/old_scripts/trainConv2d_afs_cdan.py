from tensorflow import Variable
from keras.optimizers import Adam
from adapt.feature_based import CDAN
import numpy as np
from adapt.utils import UpdateLambda

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredict import predict_afs
from src.kerasPlot import plotEncoded_afs, plotTrainingAcc, plotTrainingLoss
from src.models_v2 import getEncoder, getTask, getDiscriminator, EarlyStoppingCustom

outdir="results/afs/cdan/ghost/"
os.system(f"mkdir -p {outdir}")


# read data
source = Dataset("secondaryContact3/secondaryContact3-train.json", 1500, transpose=False, multichannel=True)
test = Dataset("secondaryContact3/secondaryContact3-test.json", 1500, transpose=False, multichannel=True)
val = Dataset("secondaryContact3/secondaryContact3-val.json", 1500, transpose=False, multichannel=True)
ghost = Dataset("ghost3/ghost3-test.json", 1500, transpose=False, multichannel=True)



# define and train cdan model
lambda_max = 20
epochs = 500
batch_size = 32
learning_rate = 0.00001
ratio_disc_enc_lr = 1


cdan = CDAN(
    #lambda_=Variable(0.0),
    lambda_ = lambda_max,
    encoder=getEncoder(shape=source.shape), 
    task=getTask(), 
    discriminator=getDiscriminator(),
    optimizer = Adam(learning_rate),
    optimizer_enc = Adam(learning_rate),
    optimizer_disc = Adam(learning_rate*ratio_disc_enc_lr),  
    loss="categorical_crossentropy",
    copy = False,
    metrics=["accuracy"],
    callbacks=[EarlyStoppingCustom()])
    #callbacks=[UpdateLambda(lambda_max=lambda_max), EarlyStoppingCustom()])

history = cdan.fit(source.afs, source.migrationStates, ghost.afs, 
                    epochs=epochs, batch_size=batch_size)


# plt training
plotTrainingAcc(cdan, "afs_training_acc_cdan.png")
plotTrainingLoss(cdan, "afs_training_loss_cdan.png")

# make predictions
np.savetxt("afs_cdan_test_cm.txt", predict_afs(cdan, test), fmt="%1.0f")
np.savetxt("afs_cdan_ghost_cm.txt", predict_afs(cdan, ghost), fmt="%1.0f")

# plot encoded space
plotEncoded_afs(cdan, source=source, target=ghost, outputpath="afs_encoded_tSNE_cdan.png")

