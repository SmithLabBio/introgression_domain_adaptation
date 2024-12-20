#!/usr/bin/env python

from tensorflow import keras
import fire
from adapt.feature_based import CDAN, MDD, ADDA
from adapt.parameter_based import FineTuning
from importlib import import_module
import json
import numpy as np
import model1 as models
import os


def run(weights_dir, data, epoch, outfile, entryname, method):
    name = os.path.splitext(os.path.basename(data))[0]
    x = np.load(data)[np.newaxis,:,:,np.newaxis]
    if method == "cdan":
        model = CDAN(
            encoder=models.getEncoder(shape=x.shape[1:]), 
            task=models.getTask(), 
            discriminator=models.getDiscriminator())
        model.fit(Xt=x, X=x, y=np.zeros((1,2)), epochs=0)
    elif method == "finetune":
        model = FineTuning(
            encoder=models.getEncoder(shape=x.shape[1:]),
            task=models.getTask())
        model.fit(x, np.zeros((1, 2)), epochs=0)
    elif method == "mdd":
        model = MDD(
            encoder=models.getEncoder(shape=x.shape[1:]), 
            task=models.getTask(), 
            discriminator=models.getDiscriminator())
        model.fit(Xt=x, X=x, y=np.zeros((1,2)), epochs=0)
    elif method == "adda":
        model = ADDA(
            encoder=models.getEncoder(shape=x.shape[1:]), 
            task=models.getTask(), 
            discriminator=models.getDiscriminator())
        model.fit(Xt=x, X=x, y=np.zeros((1,2)), epochs=0)

    model.load_weights(f"{weights_dir}/checkpoints/{epoch}.hdf5")
    pred = model.predict(x)
    with open(outfile, 'w') as fh:
        fh.write(f"{entryname},{pred[0][0]},{pred[0][1]}")

if __name__ == "__main__":
    fire.Fire(run)

