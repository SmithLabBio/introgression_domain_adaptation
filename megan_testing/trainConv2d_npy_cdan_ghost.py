from tensorflow import Variable
from keras.optimizers import Adam
from adapt.feature_based import CDAN
import numpy as np
from adapt.utils import UpdateLambda
from tensorflow.keras.utils import to_categorical
import os

from src.kerasPredict import predict
from src.kerasPlot import getEncoded, plotTrainingAcc, plotTrainingLoss
from src.models_alignments import getEncoder, getTask, getDiscriminator, EarlyStoppingCustom


outdir="results/npy/cdan/ghost/"
os.system(f"mkdir -p {outdir}")

# read data
source = np.load("secondaryContact3/secondaryContact3-train_matrices.npy")
test = np.load("secondaryContact3/secondaryContact3-test_matrices.npy")
val = np.load("secondaryContact3/secondaryContact3-val_matrices.npy")
ghost = np.load("ghost3/ghost3-test_matrices.npy")

# read labels
labels_source = to_categorical(np.load("secondaryContact3/secondaryContact3-train_labels.npy"))
labels_test = to_categorical(np.load("secondaryContact3/secondaryContact3-test_labels.npy"))
labels_val = to_categorical(np.load("secondaryContact3/secondaryContact3-val_labels.npy"))
labels_ghost = to_categorical(np.load("ghost3/ghost3-test_labels.npy"))


# define and train cdan model
lambda_max = 10
epochs = 50
batch_size = 50
learning_rate = 0.0001
ratio_disc_enc_lr = 10


cdan = CDAN(
    lambda_=Variable(0.0),
    encoder=getEncoder(shape=source.shape[1:]), 
    task=getTask(), 
    discriminator=getDiscriminator(),
    optimizer = Adam(learning_rate),
    optimizer_enc = Adam(learning_rate),
    optimizer_disc = Adam(learning_rate*ratio_disc_enc_lr),  
    loss="categorical_crossentropy",
    copy = False,
    metrics=["accuracy"],
    callbacks=[UpdateLambda(lambda_max=lambda_max), EarlyStoppingCustom()])

history = cdan.fit(source, labels_source, ghost, 
                    epochs=epochs, batch_size=batch_size)


# plt training
plotTrainingAcc(cdan, os.path.join(outdir, 'training_acc.png'))
plotTrainingLoss(cdan, os.path.join(outdir, 'training_loss.png'))

# make predictions with original network for test data and ghost data
np.savetxt(os.path.join(outdir, "test_cm.txt"), predict(cdan, test, labels_test, os.path.join(outdir, "test_roc.txt")), fmt="%1.0f")
np.savetxt(os.path.join(outdir, "ghost_cm.txt"), predict(cdan, ghost, labels_ghost, os.path.join(outdir, "ghost_roc.txt")), fmt="%1.0f")

# get encoded space
getEncoded(cdan, source=source, target=ghost, outdir = outdir, outprefix = "ghost")
getEncoded(cdan, source=source, target=test, outdir = outdir, outprefix = "test")
