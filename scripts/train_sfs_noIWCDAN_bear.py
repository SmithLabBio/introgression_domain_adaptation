#!/usr/bin/env python

# // Domain adaptation
# // Using ghost sims
# // Using optional pseudolabeling

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
from util import plot_adapt_history, save_history, stratified_subsample
import sys

from iw_cdan_v2 import IWCDAN, IWWeightUpdater


sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/"
outdir = "/mnt/scratch/smithfs/megan/da_revisions/noiwcdan_bear"

def exp(d):
    return np.format_float_scientific(d, trim='-', exp_digits=1)


def run_training(rep, max_lambda, batch, learn, enc_learn, disc_learn, target_num, target_prop, train, importance_weights=False, iw_update_every=5, iw_min_weight=0.2, iw_max_weight=5.0, force=False):
    source_path = f"{sim_dir}/bear-secondary-contact-2-20000-train-sfs-norm.npz"
    target_path = f"{sim_dir}/bear-secondary-contact-ghost-2-100-train-sfs-norm.npz"
    valid_path  = f"{sim_dir}/bear-secondary-contact-2-1000-train-sfs-norm.npz"
    subsampled_path = f"{outdir}/batch{batch}.learn_{exp(learn)}.enc_learn_{exp(enc_learn)}.disc_learn_{exp(disc_learn)}.lambda_{max_lambda}.target_{target_num}_{target_prop}_{train}_{importance_weights}/{rep}/bear-secondary-contact-ghost-2-100-train-sfs-norm.subsampled.npz"

    source = np.load(source_path) 
    target = np.load(target_path) 
    valid  = np.load(valid_path)

    # subsample target data
    subsampled_target = stratified_subsample(target, target_num, target_prop)

         
    out = f"{outdir}/batch{batch}.learn_{exp(learn)}.enc_learn_{exp(enc_learn)}.disc_learn_{exp(disc_learn)}.lambda_{max_lambda}.target_{target_num}_{target_prop}_{train}_{importance_weights}/{rep}"

    try:
        os.makedirs(out)
    except FileExistsError:
        if not force:
            quit(f"{out} already exists")
    
    np.savez(subsampled_path, data=subsampled_target)
    
    callbacks = [ModelCheckpoint(f"{out}/checkpoints/{{epoch}}.hdf5", save_weights_only=True)]
    lambda_ = Variable(0.0) 
    
    model_cls = IWCDAN if importance_weights else CDAN
    extra_kwarg = dict(n_classes=2) if importance_weights else {}


    cdan = model_cls(
        lambda_=max_lambda, # Ignore Pycharm Warning 
        encoder=models.getEncoder(shape=source["x"].shape[1:]), 
        task=models.getTask(), 
        discriminator=models.getDiscriminator(),
        optimizer = Adam(learn),
        optimizer_enc = Adam(enc_learn),
        optimizer_disc = Adam(disc_learn),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        callbacks=callbacks,
        entropy=True,
        **extra_kwarg)
    
    if importance_weights:
        cdan.set_balance_weights(source["labels"])
    
    #callbacks.append(UpdateLambda(lambda_max=max_lambda))

    if importance_weights:
        iw_callback = IWWeightUpdater(
            iwcdan_model=cdan,
            Xs_heldout = valid["x"],
            ys_heldout = valid["labels"],
            Xt=subsampled_target["x"],
            update_every=iw_update_every,
            min_weight=iw_min_weight,
            max_weight=iw_max_weight,
            momentum=0.5
        )
    
        callbacks.append(iw_callback)


    history = cdan.fit(
        X=source["x"], 
        y=to_categorical(source["labels"], 2), 
        Xt=subsampled_target["x"], 
        epochs=train, 
        batch_size=batch,
        validation_data=(valid["x"], to_categorical(valid["labels"], 2)))

    plot_adapt_history(history, out)
    save_history(history.history.history, out)

if __name__ == "__main__":
    fire.Fire(run_training)