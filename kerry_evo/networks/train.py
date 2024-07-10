#!/usr/bin/env python

from tensorflow import keras
from tensorflow import Variable
# from keras.optimizers import Adam
from keras.optimizers.legacy import Adam
from keras.callbacks import ModelCheckpoint
from adapt.feature_based import CDAN
from keras.utils import to_categorical
from adapt.utils import UpdateLambda
from scipy.spatial.distance import euclidean
import os
import json
import fire
from importlib import import_module
import numpy as np

from util import plot_adapt_history, save_history


def train(ModelFile, data_type, source_path, target_path, val_path, max_lambda, learn_rate, 
        disc_enc_learn_ratio, outdir, static_lambda=False, force=False):

    epochs = 50

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    else:
        if not force:
            quit(f"{outdir} already exists")

    source = np.load(source_path)
    target = np.load(target_path)

    if val_path:
        validation = np.load(val_path)
        validation_data = (validation["x"], to_categorical(validation["labels"], 2))
    else:
        validation_data = None

    models = import_module(ModelFile)

    config = dict(
        ModelFile=ModelFile,
        DataType=data_type,
        source_path=source_path,
        target_path=target_path,
        val_path=val_path,
        max_lambda=max_lambda,
        learn_rate=learn_rate,
        disc_enc_learn_ratio=disc_enc_learn_ratio,
        outdir=outdir)
    with open(f"{outdir}/config.json", 'w') as fh:
        json.dump(config, fh)
    
    callbacks = [ModelCheckpoint(f"{outdir}/checkpoints/{{epoch}}.hdf5", save_weights_only=True)]

    if not static_lambda:
        lambda_ = Variable(0.0) 
        callbacks.append(UpdateLambda(lambda_max=max_lambda))
    else:
        lambda_ = max_lambda

    model = CDAN(
        lambda_=lambda_, # Ignore Pycharm Warning
        encoder=models.getEncoder(shape=source["x"].shape[1:]), 
        task=models.getTask(), 
        discriminator=models.getDiscriminator(),
        optimizer = Adam(learn_rate),
        optimizer_enc = Adam(learn_rate),
        optimizer_disc = Adam(learn_rate * disc_enc_learn_ratio),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        callbacks=callbacks)


    history = model.fit(
        X=source["x"], 
        y=to_categorical(source["labels"], 2), 
        Xt=target["x"], 
        epochs=epochs, 
        batch_size=32, 
        validation_data=validation_data)

    plot_adapt_history(history, outdir)
    save_history(history.history.history, outdir)

if __name__ == "__main__":
    fire.Fire(train)





