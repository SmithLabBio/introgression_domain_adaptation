#!/usr/bin/env python3

import fire
from pydantic import BaseModel
from typing import Tuple 
import oyaml as yaml
import pickle
import os.path as path
import msprime as mp
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy

class Config(BaseModel):
    seed: int
    nDatasets: int
    nSamples: int
    sequenceLength: int
    recombinationRate: float
    mutationRate: float
    migrationRateRange: Tuple[float, float]
    initialSizeA: int
    initialSizeB: int
    initialSizeC: int
    splitTime: int 

def simulate(nDatasets, configPath, outPath= "", seed=None, force=False) -> None: 
    if not outPath:
        outPath = f"{path.splitext(configPath)[0]}.npz"
    else:
        if not path.splitext(outPath)[1] == ".npz":
            quit(f"Aborted: outPath should have extension \".npz\"")
    if path.exists(outPath): 
        if not force:
            quit(f"Aborted: {outPath} already exists")
    if seed: 
        np.random.seed(seed)
    else:
        seed = int(np.random.rand() * (2**32 - 1))
        np.random.seed(seed)

    config = Config(**yaml.safe_load(open(configPath)), seed=seed, nDatasets=nDatasets)

    # Random seeds
    ancestrySeeds = np.random.randint(2**32, size=config.nDatasets)
    mutationSeeds = np.random.randint(2**32, size=config.nDatasets)

    # Migration rates
    migrationRates = np.random.uniform(low=config.migrationRateRange[0], 
                                       high=config.migrationRateRange[1], 
                                       size=config.nDatasets)

    # Initialize output arrays
    positions = np.empty(config.nDatasets, dtype=object) 
    charMatrices = np.empty(config.nDatasets, dtype=object) 
    statDict = dict(all=[], pop1=[], pop2=[])
    summaryStats = dict(dxy=[], fst=[], 
            # pi=dict(all=[], pop1=[], pop2=[]), 
            pi=statDict.copy(), 
            # tajimasD=dict(all=[], pop1=[], pop2=[]), 
            tajimasD=statDict.copy(), 
            # segregatingSites=dict(all=[], pop1=[], pop2=[]),
            segregatingSites=statDict.copy(),
            # sfs=dict(all=[], pop1=[], pop2=[])) 
            sfs=statDict.copy()) 

    # Simulate data
    for i in tqdm(range(config.nDatasets), desc="Simulating data"):
        # Build demographic model
        dem = mp.Demography()
        dem.add_population(name="a", initial_size=config.initialSizeA)
        dem.add_population(name="b", initial_size=config.initialSizeB)
        dem.add_population(name="c", initial_size=config.initialSizeC)
        dem.add_population_split(time=config.splitTime, derived=["b", "c"], ancestral="a")
        dem.set_symmetric_migration_rate(["b", "c"], rate=migrationRates[i])
        # Simulate ancestry for samples
        ts = mp.sim_ancestry(samples={"b": config.nSamples, "c": config.nSamples},
                demography=dem, random_seed=ancestrySeeds[i], 
                sequence_length=config.sequenceLength, 
                recombination_rate=config.recombinationRate)

        # Simulate mutations for ancestries
        mts = mp.sim_mutations(ts, rate=config.mutationRate, random_seed=mutationSeeds[i])
        positions[i] = mts.tables.sites.position.astype(np.int32)
        charMatrices[i] = mts.genotype_matrix().astype(np.uint8).transpose() # Node: Consumes a lot of memory, shape: [sites, samples] (before transpose)


        # # Compute and store summary stats
        summaryStats["dxy"].append(mts.divergence(sample_sets=[ts.samples(1), ts.samples(2)]))
        summaryStats["fst"].append(mts.Fst(sample_sets=[ts.samples(1), ts.samples(2)]))

        def statAppend(stat, values):
            summaryStats[stat]["all"].append(values[0])
            summaryStats[stat]["pop1"].append(values[1])
            summaryStats[stat]["pop2"].append(values[2])

        statAppend("pi", mts.diversity(sample_sets=[ts.samples(), ts.samples(1), ts.samples(2)]))
        statAppend("tajimasD", mts.Tajimas_D(sample_sets=[ts.samples(), ts.samples(1), ts.samples(2)]))
        statAppend("segregatingSites", mts.segregating_sites(sample_sets=[ts.samples(), ts.samples(1), ts.samples(2)]))
        statAppend("sfs", mts.allele_frequency_spectrum(sample_sets=[ts.samples(), ts.samples(1), ts.samples(2)]))

    # Write output to file
    print("Writing data ...")
    data = dict(
        config=pickle.dumps(config.model_dump()),
        migrationRates=migrationRates,
        positions=positions,
        charMatrices=charMatrices,
        summaryStats=pickle.dumps(summaryStats))

    np.savez_compressed(outPath, **data)
    print("Simulations Complete!")

if __name__ == "__main__":
    fire.Fire(simulate)
