#!/usr/bin/env python

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
# outdir = "/mnt/scratch/smithfs/cobb/popai/bear2"
outdir = "/mnt/scratch/smithfs/cobb/popai/bear3"

def run_training(meth, format, rep, batch, learn, enc_learn, epochs):
    if format == "sfs":
        source_path = f"{sim_dir}/bear-secondary-contact-1-20000-train-sfs.npz"
        target_path = f"{sim_dir}/bear-secondary-contact-ghost-1-1000-train-sfs.npz"
        valid_path  = f"{sim_dir}/bear-secondary-contact-1-1000-train-sfs.npz"
    elif format == "norm":
        source_path = f"{sim_dir}/bear-secondary-contact-1-20000-train-sfs-norm.npz"
        target_path = f"{sim_dir}/bear-secondary-contact-ghost-1-1000-train-sfs-norm.npz"
        valid_path  = f"{sim_dir}/bear-secondary-contact-1-1000-train-sfs-norm.npz"

    source = np.load(source_path) 
    target = np.load(target_path) 
    valid  = np.load(valid_path) 

    def exp(d):
        return np.format_float_scientific(d, trim='-', exp_digits=1)
         
    out_prefix = f"{outdir}/batch{batch}.learn-{exp(learn)}.enc_learn-{exp(enc_learn)}"
    out_suffix = f"{format}/{rep}"
    if meth == "finetune":
        finetune_out = f"{out_prefix}.finetune.{out_suffix}"
        os.makedirs(finetune_out)
        finetune_callbacks = [ModelCheckpoint(f"{finetune_out}/checkpoints/{{epoch}}.hdf5", save_weights_only=True)]
        finetune = FineTuning(encoder=models.getEncoder(shape=source["x"].shape[1:]),
            task=models.getTask(),
            optimizer=Adam(learn),
            optimizer_enc=Adam(enc_learn),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            callbacks=finetune_callbacks)
        finetune_history = finetune.fit(
            source["x"], 
            to_categorical(source["labels"], 2), 
            epochs=epochs, 
            batch_size=batch,
            validation_data=(valid["x"], to_categorical(valid["labels"], 2)))
        plot_adapt_history(finetune_history, finetune_out)
        save_history(finetune_history.history.history, finetune_out)

    elif meth == "cdan":
        cdan_out = f"{out_prefix}-cdan-{out_suffix}"
        os.makedirs(cdan_out)
        cdan_callbacks = [ModelCheckpoint(f"{cdan_out}/checkpoints/{{epoch}}.hdf5", save_weights_only=True)]
        cdan = CDAN(
            lambda_=0, 
            encoder=models.getEncoder(shape=source["x"].shape[1:]), 
            task=models.getTask(), 
            discriminator=models.getDiscriminator(),
            optimizer = Adam(learn),
            optimizer_enc = Adam(enc_learn),
            optimizer_disc = Adam(0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            callbacks=cdan_callbacks)
        cdan_history = cdan.fit(
            X=source["x"], 
            y=to_categorical(source["labels"], 2), 
            Xt=target["x"], 
            epochs=epochs, 
            batch_size=batch,
            validation_data=(valid["x"], to_categorical(valid["labels"], 2)))
        plot_adapt_history(cdan_history, cdan_out, disc=False)
        save_history(cdan_history.history.history, cdan_out)

if __name__ == "__main__":
    fire.Fire(run_training)