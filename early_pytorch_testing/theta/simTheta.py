#!/usr/bin/env python3

import fire
import pickle
import os.path as path
import msprime as mp
import numpy as np
from tqdm import tqdm

def simulate(nDatasets, outPrefix, seed=None, force=False) -> None: 
    outPath = f"{outPrefix}.pickle"
    if path.exists(outPath): 
        if not force:
            quit(f"Aborted: {outPath} already exists")
    if seed: 
        np.random.seed(seed)
    else:
        seed = int(np.random.rand() * (2**32 - 1))
        np.random.seed(seed)
    
    nSamples = 50 
    seqLength = 100000
    recombRate = 1e-8 
    mutRate = 1e-6
    lowerSize = 1000 
    upperSize = 100000 

    # Random seeds
    ancestrySeeds = np.random.randint(2**32, size=nDatasets)
    mutationSeeds = np.random.randint(2**32, size=nDatasets)

    popSizes = np.random.randint(lowerSize, upperSize, size=nDatasets)
    charMatrices = []

    for i in tqdm(range(nDatasets), desc="Simulating data"):
        ts = mp.sim_ancestry(samples=nSamples, recombination_rate=recombRate, 
                sequence_length=seqLength, population_size=popSizes[i], 
                random_seed=ancestrySeeds[i])
        mts = mp.sim_mutations(ts, rate=mutRate, random_seed=mutationSeeds[i])
        snps = mts.genotype_matrix().transpose() # Transpose to put haplotypes first
        charMatrices.append(snps)  

    print(f"Writing output ...")
    data = dict(nDatasets=nDatasets, nSamples=nSamples, popSizes=popSizes, 
                charMatrices=charMatrices)
    with open(outPath, "wb") as fh:
        pickle.dump(data, fh)

    print("Simulations Complete!")

if __name__ == "__main__":
    fire.Fire(simulate)
