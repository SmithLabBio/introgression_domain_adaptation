from keras.optimizers import Adam
from adapt.parameter_based import FineTuning
import numpy as np
import pandas as pd

from src.kerasPredict import predict
from src.kerasPlot import getEncoded
from src.models_afs import getEncoder, getTask
from tensorflow.keras.utils import to_categorical
import os

outdir="results/afs/original/"
os.system(f"mkdir -p {outdir}")

# read data
source = np.load("secondaryContact3/secondaryContact3-train_afs.npy")
test = np.load("secondaryContact3/secondaryContact3-test_afs.npy")
val = np.load("secondaryContact3/secondaryContact3-val_afs.npy")
ghost = np.load("ghost3/ghost3-test_afs.npy")
bgs = np.load("bgs/bgs-test_afs.npy")

# expand dims
source = np.expand_dims(source, axis=-1)
test = np.expand_dims(test, axis=-1)
val = np.expand_dims(val, axis=-1)
ghost = np.expand_dims(ghost, axis=-1)
bgs = np.expand_dims(bgs, axis=-1)

# read labels
labels_source = to_categorical(np.load("secondaryContact3/secondaryContact3-train_labels.npy"))
labels_test = to_categorical(np.load("secondaryContact3/secondaryContact3-test_labels.npy"))
labels_val = to_categorical(np.load("secondaryContact3/secondaryContact3-val_labels.npy"))
labels_ghost = to_categorical(np.load("ghost3/ghost3-test_labels.npy"))
labels_bgs = to_categorical(np.load("bgs/bgs-test_labels.npy"))

# train the original model
learning_rate = 0.0001
epochs = 10
batch_size = 32

finetunig = FineTuning(encoder=getEncoder(shape=source.shape[1:]),
                         task=getTask(),
                         optimizer=Adam(learning_rate),
                         optimizer_enc=Adam(learning_rate),
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])
history_finetunig = finetunig.fit(source, labels_source, epochs = epochs, batch_size = batch_size, validation_data=(val, labels_val))

# make predictions with original network for test data and ghost data
np.savetxt(os.path.join(outdir, "test_cm.txt"), predict(finetunig, test, labels_test, os.path.join(outdir, "test_roc.txt")), fmt="%1.0f")
np.savetxt(os.path.join(outdir, "bgs_cm.txt"), predict(finetunig, bgs, labels_bgs, os.path.join(outdir, "bgs_roc.txt")), fmt="%1.0f")
np.savetxt(os.path.join(outdir, "ghost_cm.txt"), predict(finetunig, ghost, labels_ghost, os.path.join(outdir, "ghost_roc.txt")), fmt="%1.0f")


# get encoded space
getEncoded(finetunig, source=source, target=ghost, outdir = outdir, outprefix = "ghost")
getEncoded(finetunig, source=source, target=test, outdir = outdir, outprefix = "test")
getEncoded(finetunig, source=source, target=bgs, outdir = outdir, outprefix = "bgs")

