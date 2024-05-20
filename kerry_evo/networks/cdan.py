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

from simulations.secondary_contact import SecondaryContact
from simulations.secondary_contact_ghost import GhostSecondaryContact
from sim_wrapper.numpy_dataset import NumpySnpDataset, NumpyAfsDataset

from util import plot_adapt_history, save_history
# from models import getEncoder, getTask, getDiscriminator


def train(ModelFile, DataType, SrcType, TgtType, source_path, target_path, val_path, n_snps,
          max_lambda, learn_rate, disc_enc_learn_ratio, outdir):

    epochs = 50

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    else:
        quit(f"{outdir} already exists")

    match DataType:
        case "NumpySnpDataset": 
            source =     NumpySnpDataset(eval(SrcType), source_path, "migration_state", n_snps=n_snps, split=True, sorting=euclidean)
            target =     NumpySnpDataset(eval(TgtType), target_path, "migration_state", n_snps=n_snps, split=True, sorting=euclidean)
            validation = NumpySnpDataset(eval(TgtType), val_path,    "migration_state", n_snps=n_snps, split=True, sorting=euclidean)
        case "NumpyAfsDataset":
            source =     NumpyAfsDataset(eval(SrcType), source_path, "migration_state", expand_dims=True)
            target =     NumpyAfsDataset(eval(TgtType), target_path, "migration_state", expand_dims=True)
            validation = NumpyAfsDataset(eval(TgtType), val_path,    "migration_state", expand_dims=True)
        case _: 
            quit("Invalid DataType argument")

    models = import_module(ModelFile)

    config = dict(
        ModelFile=ModelFile,
        DataType=DataType,
        SrcType=SrcType,
        TgtType=TgtType,
        source_path=source_path,
        target_path=target_path,
        val_path=val_path,
        n_snps=n_snps,
        max_lambda=max_lambda,
        learn_rate=learn_rate,
        disc_enc_learn_ratio=disc_enc_learn_ratio,
        outdir=outdir)
    with open(f"{outdir}/config.json", 'w') as fh:
        json.dump(config, fh)
    
    model = CDAN(
        lambda_=Variable(0.0), # Ignore Pycharm Warning
        encoder=models.getEncoder(shape=source.x.shape[1:]), 
        task=models.getTask(), 
        discriminator=models.getDiscriminator(),
        optimizer = Adam(learn_rate),
        optimizer_enc = Adam(learn_rate),
        optimizer_disc = Adam(learn_rate * disc_enc_learn_ratio),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        callbacks=[
            ModelCheckpoint(f"{outdir}/checkpoints/{{epoch}}.hdf5", save_weights_only=True), 
            UpdateLambda(lambda_max=max_lambda)])

    history = model.fit(
        X=source.x, 
        y=to_categorical(source.labels, 2), 
        Xt=target.x, 
        epochs=epochs, 
        batch_size=64, 
        validation_data=(validation.x, to_categorical(validation.labels, 2)))

    plot_adapt_history(history, outdir)
    save_history(history.history.history, outdir)


if __name__ == "__main__":
    fire.Fire(train)





