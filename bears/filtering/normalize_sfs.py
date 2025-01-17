#!/usr/bin/env python

import fire
import os
import numpy as np


def normalize(path, out):
    if path.endswith(".npz"): 
        d = np.load(path)
        m = d["x"]
        # Get sum of each row in the matrix
        sums = np.sum(m, axis=(1,2,3), keepdims=False)
        # Divide the values of each row by their sum
        normalized = m / sums
        np.savez(out, x=normalized, labels=d["labels"])
    elif path.endswith(".npy"):
        m = np.load(path)
        sum = np.sum(m)
        normalized = m / sum
        np.save(out, normalized)

if __name__ == "__main__":
    fire.Fire(normalize)

