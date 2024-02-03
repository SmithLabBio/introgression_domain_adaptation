#!/usr/bin/env python

import fire
import numpy as np
import pickle

def run(path):
    """Print meta data associated with simulated data"""
    data = np.load(path, allow_pickle=True)
    for k, v in pickle.loads(data["config"]).items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    fire.Fire(run)