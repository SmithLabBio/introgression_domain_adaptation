#!/usr/bin/env python

# // Domain Adaptation with bears using all chromosome for each population 
# // Using ghost 1 simulations

from tensorflow import keras
from tensorflow import Variable
# from keras.optimizers import Adam
from keras.optimizers.legacy import Adam
from keras.callbacks import ModelCheckpoint
from adapt.feature_based import CDAN
from adapt.parameter_based import FineTuning
from keras.utils import to_categorical
from adapt.utils import UpdateLambda
from scipy.spatial.distance import euclidean
import os
import json
import fire
from importlib import import_module
import numpy as np

import model1 as models
from util import plot_adapt_history, save_history

sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations"
outdir = "/mnt/scratch/smithfs/cobb/popai/bear4"

def run_training(rep, max_lambda, batch, learn, enc_learn, disc_learn, epochs):
    source_path = f"{sim_dir}/bear-secondary-contact-1-20000-train-sfs-norm.npz"
    target_path = f"{sim_dir}/bear-secondary-contact-ghost-1-100-train-sfs-norm.npz"
    valid_path  = f"{sim_dir}/bear-secondary-contact-1-1000-train-sfs-norm.npz"

    source = np.load(source_path) 
    target = np.load(target_path) 
    valid  = np.load(valid_path) 

    def exp(d):
        return np.format_float_scientific(d, trim='-', exp_digits=1)
         
    out = f"{outdir}/batch{batch}.learn_{exp(learn)}.enc_learn_{exp(enc_learn)}.disc_learn_{exp(disc_learn)}.lambda_{max_lambda}/{rep}"

    os.makedirs(out)
    callbacks = [ModelCheckpoint(f"{out}/checkpoints/{{epoch}}.hdf5", save_weights_only=True)]
    lambda_ = Variable(0.0) 
    callbacks.append(UpdateLambda(lambda_max=max_lambda))
    cdan = CDAN(
        lambda_=lambda_, # Ignore Pycharm Warning 
        encoder=models.getEncoder(shape=source["x"].shape[1:]), 
        task=models.getTask(), 
        discriminator=models.getDiscriminator(),
        optimizer = Adam(learn),
        optimizer_enc = Adam(enc_learn),
        optimizer_disc = Adam(disc_learn),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        callbacks=callbacks)
    history = cdan.fit(
        X=source["x"], 
        y=to_categorical(source["labels"], 2), 
        Xt=target["x"], 
        epochs=epochs, 
        batch_size=batch,
        validation_data=(valid["x"], to_categorical(valid["labels"], 2)))
    plot_adapt_history(history, out)
    save_history(history.history.history, out)

if __name__ == "__main__":
    fire.Fire(run_training)