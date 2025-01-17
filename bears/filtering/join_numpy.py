#!/usr/bin/env python

import fire
import numpy as np

from sim_wrapper.numpy_dataset import NumpyAfsDataset

def join(outpath, *paths): 
    arrays = []
    for i in paths:
        arrays.append(np.load(i))
    dataset = NumpyAfsDataset.from_arrays(x=np.array(arrays), expand_dims=True, labels=None)
    dataset.save(outpath)
    print(dataset.x)

if __name__ == "__main__":
    fire.Fire(join)