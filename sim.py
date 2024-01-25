#!/usr/bin/env python3

import fire
from pydantic import BaseModel
from typing import Tuple 
import oyaml as yaml
import pickle
import os.path as path
from progress.bar import Bar
import msprime as mp
import numpy as np

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

def simulate(configPath: str, outDir: str = ".") -> None: 
    config = Config(**yaml.safe_load(open(configPath)))

    # Random seeds
    rng = np.random.default_rng(config.seed)
    ancestrySeeds = rng.integers(2**32, size=config.nDatasets)
    mutationSeeds = rng.integers(2**32, size=config.nDatasets)

    # Migration rates
    migrationRates = rng.uniform(low=config.migrationRateRange[0], 
            high=config.migrationRateRange[1], size=config.nDatasets)

    # Initialize output arrays
    positions = np.empty(config.nDatasets, dtype=object) 
    charMatrices = np.empty(config.nDatasets, dtype=object) 

    # Simulate data
    with Bar("Simulating data", max=config.nDatasets) as bar:
        for i in range(config.nDatasets):

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
            positions[i] = mts.tables.sites.position.astype(np.int64)
            charMatrices[i] = mts.genotype_matrix() # Node: Consumes a lot of memory

            bar.next()

    # Write output to file
    print("Writing data ...")
    data = dict(
        config=pickle.dumps(config.model_dump()),
        ancesrtySeeds=ancestrySeeds,
        mutationSeeds=mutationSeeds,
        migrationRates=migrationRates,
        positions=positions,
        charMatrices=charMatrices)
    outfile = f"{path.splitext(path.basename(configPath))[0]}.npz"
    outpath = path.join(outDir, outfile)
    np.savez_compressed(outpath, **data)

if __name__ == "__main__":
    fire.Fire(simulate)
