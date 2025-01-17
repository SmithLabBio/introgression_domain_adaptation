#!/usr/bin/env python

import os
import fire
import numpy as np
from scipy.spatial.distance import euclidean

from simulations.secondary_contact import SecondaryContact
from simulations.secondary_contact_ghost2 import GhostSecondaryContact
from sim_wrapper.numpy_dataset import NumpySnpDataset, NumpyAfsDataset


def convert(DataType, SimType, inpath, suffix, n_snps=None, force=False, polarized=False, normalized=False):
    outpath = f"{os.path.splitext(inpath)[0]}-{suffix}.npz"
    if os.path.exists(outpath):
        if not force:
            quit(f"{outpath} already exists")

    match DataType:
        case "NumpySnpDataset": 
            data =     NumpySnpDataset(eval(SimType), inpath, "migration_state", n_snps=n_snps, split=True, sorting=euclidean)
        case "NumpyAfsDataset":
            data =     NumpyAfsDataset(eval(SimType), inpath, "migration_state", expand_dims=True, polarized=False, normalized=normalized)
        case _: 
            quit("Invalid DataType argument")

    np.savez(outpath, x=data.x, labels=data.labels)

if __name__ == "__main__":
    fire.Fire(convert)





