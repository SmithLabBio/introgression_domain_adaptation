from keras.optimizers import Adam
from adapt.parameter_based import FineTuning
import numpy as np
from tensorflow.keras.utils import to_categorical

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredict import predict_npy
from src.kerasPlot import plotEncoded_npy
from src.models_v2 import getEncoder, getTask
import os

outdir="results/npy/original/"
os.system(f"mkdir -p {outdir}")

# read data
source = np.load("secondaryContact3/secondaryContact3-train_matrices.npy")
test = np.load("secondaryContact3/secondaryContact3-test_matrices.npy")
val = np.load("secondaryContact3/secondaryContact3-val_matrices.npy")
ghost = np.load("ghost3/ghost3-test_matrices.npy")
bgs = np.load("bgs/bgs-test_matrices.npy")

# read labels
labels_source = to_categorical(np.load("secondaryContact3/secondaryContact3-train_labels.npy"))
labels_test = to_categorical(np.load("secondaryContact3/secondaryContact3-test_labels.npy"))
labels_val = to_categorical(np.load("secondaryContact3/secondaryContact3-val_labels.npy"))
labels_ghost = to_categorical(np.load("ghost3/ghost3-test_labels.npy"))
labels_bgs = to_categorical(np.load("bgs/bgs-test_labels.npy"))

# train the original model
learning_rate = 0.0001
epochs = 5
batch_size = 32

finetunig = FineTuning(encoder=getEncoder(shape=source.shape[1:]),
                         task=getTask(),
                         optimizer=Adam(learning_rate),
                         optimizer_enc=Adam(learning_rate),
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])
history_finetunig = finetunig.fit(source, labels_source, epochs = epochs, batch_size = batch_size, validation_data=(val, labels_val))

# make predictions with original network for test data and ghost data
np.savetxt(os.path.join(outdir, "test_cm.txt"), predict_npy(finetunig, test, labels_test, os.path.join(outdir, "test_roc.txt")), fmt="%1.0f")
np.savetxt(os.path.join(outdir, "ghost_cm.txt"), predict_npy(finetunig, ghost, labels_ghost, os.path.join(outdir, "ghost_roc.txt")), fmt="%1.0f")
np.savetxt(os.path.join(outdir, "bgs_cm.txt"), predict_npy(finetunig, bgs, labels_bgs, os.path.join(outdir, "bgs_roc.txt")), fmt="%1.0f")


# plot encoded space for original
plotEncoded_npy(finetunig, source=source, target=ghost, outdir = outdir, outprefix = "ghost")
plotEncoded_npy(finetunig, source=source, target=test, outdir = outdir, outprefix = "test")
plotEncoded_npy(finetunig, source=source, target=bgs, outdir = outdir, outprefix = "bgs")
