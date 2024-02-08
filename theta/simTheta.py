#!/usr/bin/env python3

import fire
import msprime
import numpy as np
import pickle
import yaml
import os.path as p
from tqdm import tqdm


def sim(seed, targetDatasetNum, path):
    nSamples = 40 
    seqLength = 10000
    recombRate = 1e-8 
    mutRate = 1e-6
    lowerSize = 1000 
    upperSize = 100000 

    popSizes = [] 
    charMatrices = []
    nDatasets = 0

    for i in tqdm(range(targetDatasetNum), desc="Simulating data"):
        popSize = np.random.randint(lowerSize, upperSize)
        popSizes.append(popSize)
        ts = msprime.sim_ancestry(samples=nSamples, recombination_rate=recombRate, 
                    sequence_length=seqLength, population_size=popSize)
        mts = msprime.sim_mutations(ts, rate=mutRate, random_seed=seed, 
            model=msprime.BinaryMutationModel())
        snps = mts.genotype_matrix().transpose()
        if snps.shape[1] >= 500:
            charMatrices.append(snps) # Transpose to put haplotypes first 
            nDatasets += 1

    print(f"Saving {nDatasets} datasets")
    data = dict(nDatasets=nDatasets, nSamples=nSamples, popSizes=popSizes, 
                charMatrices=charMatrices)
    with open(path, "wb") as fh:
        pickle.dump(data, fh)

sim(1234, 1000, "theta-sims.pickle")
sim(1235, 100, "theta-sims-test.pickle")

print("Simulations complete")
