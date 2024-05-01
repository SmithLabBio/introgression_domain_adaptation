from tensorflow import Variable
from keras.optimizers import Adam
from adapt.feature_based import CDAN
import numpy as np
from adapt.utils import UpdateLambda
import os

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredict import predict_afs_npy
from src.kerasPlot import plotEncoded_npy, plotTrainingAcc, plotTrainingLoss
from src.models_afs_npy import getEncoder, getTask, getDiscriminator, EarlyStoppingCustom
from tensorflow.keras.utils import to_categorical


outdir="results/afs/cdan/bgs/"
os.system(f"mkdir -p {outdir}")

# read data
source = np.load("secondaryContact3/secondaryContact3-train_afs.npy")
test = np.load("secondaryContact3/secondaryContact3-test_afs.npy")
val = np.load("secondaryContact3/secondaryContact3-val_afs.npy")
bgs = np.load("bgs/bgs-test_afs.npy")

# expand dims
source = np.expand_dims(source, axis=-1)
test = np.expand_dims(test, axis=-1)
val = np.expand_dims(val, axis=-1)
bgs = np.expand_dims(bgs, axis=-1)

# read labels
labels_source = to_categorical(np.load("secondaryContact3/secondaryContact3-train_labels.npy"))
labels_test = to_categorical(np.load("secondaryContact3/secondaryContact3-test_labels.npy"))
labels_val = to_categorical(np.load("secondaryContact3/secondaryContact3-val_labels.npy"))
labels_bgs = to_categorical(np.load("bgs/bgs-test_labels.npy"))



# define and train cdan model
lambda_max = 50
epochs = 100
batch_size = 32
learning_rate = 0.0001
ratio_disc_enc_lr = 1


cdan = CDAN(
    #lambda_=Variable(0.0),
    lambda_ = lambda_max,
    encoder=getEncoder(shape=source.shape[1:]), 
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

history = cdan.fit(source, labels_source, bgs, 
                    epochs=epochs, batch_size=batch_size)


# plt training
plotTrainingAcc(cdan, os.path.join(outdir, 'training_acc.png'))
plotTrainingLoss(cdan, os.path.join(outdir, 'training_loss.png'))

# make predictions with original network for test data and bgs data
np.savetxt(os.path.join(outdir, "test_cm.txt"), predict_afs_npy(cdan, test, labels_test, os.path.join(outdir, "test_roc.txt")), fmt="%1.0f")
np.savetxt(os.path.join(outdir, "bgs_cm.txt"), predict_afs_npy(cdan, bgs, labels_bgs, os.path.join(outdir, "bgs_roc.txt")), fmt="%1.0f")


# plot encoded space for original
plotEncoded_npy(cdan, source=source, target=bgs, outdir = outdir, outprefix = "bgs")

# plot encoded space for original
plotEncoded_npy(cdan, source=source, target=test, outdir = outdir, outprefix = "test")
