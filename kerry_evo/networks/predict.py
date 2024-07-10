#!/usr/bin/env python

from tensorflow import keras
import fire
from adapt.feature_based import CDAN
from importlib import import_module
import json
import numpy as np


def run(data, weights, config, outpath):
    with open(config) as fh:
        config = json.load(fh)

    x = np.load(data)[np.newaxis,:,:,np.newaxis]
    models = import_module(config["ModelFile"]) 

    model = CDAN(
        encoder=models.getEncoder(shape=x.shape[1:]), 
        task=models.getTask(), 
        discriminator=models.getDiscriminator())

    model.fit(Xt=x, X=x, y=np.zeros((1,2)), epochs=0)
    model.load_weights(weights)
    pred = model.predict(x)
    np.save(outpath, pred[0])

if __name__ == "__main__":
    fire.Fire(run)

