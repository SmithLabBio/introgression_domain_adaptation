#!/usr/bin/env python

from tensorflow import keras
from tensorflow import Variable
# from keras.optimizers import Adam
from keras.optimizers.legacy import Adam
from keras.callbacks import ModelCheckpoint
from adapt.feature_based import ADDA 
from keras.utils import to_categorical
from adapt.utils import UpdateLambda
from scipy.spatial.distance import euclidean
import os
import json
import fire
from importlib import import_module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import model1 as models
from util import plot_adapt_history, save_history

sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations"
bear_dir = "/mnt/scratch/smithlab/cobb/bears/filtered"
outdir = "/mnt/scratch/smithfs/cobb/popai/bear_real3"

def run_training(pops, rep, batch, learn, enc_learn, disc_learn, epochs):
    source_path = f"{sim_dir}/bear-secondary-contact-1-20000-train-sfs-norm.npz"
    target_path = f"{bear_dir}/{pops}_norm.npz"
    valid_path  = f"{sim_dir}/bear-secondary-contact-1-1000-train-sfs-norm.npz"

    source = np.load(source_path) 
    target = np.load(target_path) 
    valid  = np.load(valid_path) 

    def exp(d):
        return np.format_float_scientific(d, trim='-', exp_digits=1)
         
    out = f"{outdir}/batch{batch}.learn_{exp(learn)}.enc_learn_{exp(enc_learn)}.disc_learn_{exp(disc_learn)}/{pops}/{rep}"

    if os.path.exists(out):
        print("Output directory already exists.")
        quit(0)

    os.makedirs(out)
    callbacks = [ModelCheckpoint(f"{out}/checkpoints/{{epoch}}.hdf5", save_weights_only=True)]
    adda = ADDA(
        encoder=models.getEncoder(shape=source["x"].shape[1:]), 
        task=models.getTask(), 
        discriminator=models.getDiscriminator(),
        optimizer = Adam(learn),
        optimizer_enc = Adam(enc_learn),
        optimizer_disc = Adam(disc_learn),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        callbacks=callbacks)
    history = adda.fit(
        X=source["x"], 
        y=to_categorical(source["labels"], 2), 
        Xt=target["x"], 
        epochs=epochs, 
        batch_size=batch,
        validation_data=(valid["x"], to_categorical(valid["labels"], 2)))

    # plot_adapt_history(history, out)
    pd.DataFrame(adda.history_).plot(figsize=(8,5))
    plt.title("Training history", fontsize=14); plt.xlabel("Epochs"); plt.ylabel("Scores")
    plt.legend(ncol=2)
    plt.savefig(f"{out}/loss-acc.png", bbox_inches="tight")
    save_history(history.history.history, out)

if __name__ == "__main__":
    fire.Fire(run_training)