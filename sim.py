#!/usr/bin/env python3

import msprime as mp
# import tskit as tk
import numpy as np
# import demesdraw as dd
# import matplotlib.pyplot as plt
import fire
import oyaml as yaml
from schema import Schema, And, Use, Optional, SchemaError
import oyaml as yaml
import pickle
import os.path as path
from dotmap import DotMap
from progress.bar import Bar


# Schema for input configuration file

def intSci(s):
    """Convert integer string written in scientific notation to int"""
    return int(float(s))

YAML_SCHEMA = Schema({
    "seed": And(int),
    "nDatasets": And(Use(intSci)),
    "nSamples": And(Use(intSci)),
    "sequenceLength": And(Use(intSci)),
    "recombinationRate": And(Use(float)),
    "mutationRate": And(Use(float)),
    "migrationRateRange": And( [Use(float)], lambda a: len(a) == 2 ),
    "initialSizeA": And(Use(intSci)),
    "initialSizeB": And(Use(intSci)),
    "initialSizeC": And(Use(intSci)),
    "splitTime": And(Use(intSci))
})

class TwoPopSimulation():
    def __init__(self, configPath: str) -> None:
        self.configPath = configPath
        # Parse configuration file
        with open(configPath, "r") as f:
            unvalidatedConfig = yaml.safe_load(f)
        self.config = DotMap(YAML_SCHEMA.validate(unvalidatedConfig))
        # Random seeds
        rng = np.random.default_rng(self.config.seed)
        self.ancestrySeeds = rng.integers(2**32, size=self.config.nDatasets)
        self.mutationSeeds = rng.integers(2**32, size=self.config.nDatasets)
        # Migration rates
        self.migrationRates = rng.uniform(low=self.config.migrationRateRange[0], 
                high=self.config.migrationRateRange[1], size=self.config.nDatasets)
        # Initialize output arrays
        self.positions = np.empty(self.config.nDatasets, dtype=object) 
        self.charMatrices = np.empty(self.config.nDatasets, dtype=object) 

    def runReplicate(self, it: int):
        # Build demographic model
        dem = mp.Demography()
        dem.add_population(name="a", initial_size=self.config.initialSizeA)
        dem.add_population(name="b", initial_size=self.config.initialSizeB)
        dem.add_population(name="c", initial_size=self.config.initialSizeC)
        dem.add_population_split(time=self.config.splitTime, derived=["b", "c"], ancestral="a")
        dem.set_symmetric_migration_rate(["b", "c"], rate=self.migrationRates[it])
        # Simulate ancestry for samples
        ts = mp.sim_ancestry(samples={"b": self.config.nSamples, "c": self.config.nSamples},
                demography=dem, random_seed=self.ancestrySeeds[it], 
                sequence_length=self.config.sequenceLength, 
                recombination_rate=self.config.recombinationRate)
        # Simulate mutations for ancestries
        mts = mp.sim_mutations(ts, rate=self.config.mutationRate, random_seed=self.mutationSeeds[it])
        self.positions[it] = mts.tables.sites.position.astype(np.int64)
        self.charMatrices[it] = mts.genotype_matrix() # Consumes a lot of memory

    def write(self, outDir): 
        print("Writing data ...")
        data = dict(
            config=pickle.dumps(self.config.toDict()),
            ancesrtySeeds=self.ancestrySeeds,
            mutationSeeds=self.mutationSeeds,
            migrationRates=self.migrationRates,
            positions=self.positions,
            charMatrices=self.charMatrices)
        outfile = f"{path.splitext(path.basename(self.configPath))[0]}.npz"
        outpath = path.join(outDir, outfile)
        np.savez_compressed(outpath, **data)

def simulate(configPath: str, outDir: str = ".") -> None:
    sim = TwoPopSimulation(configPath)
    with Bar("Simulating data", max=sim.config.nDatasets) as bar:
        for i in range(sim.config.nDatasets):
            sim.runReplicate(i)
            bar.next()
    sim.write(outDir) 
    print("Simulation Complete")

if __name__ == "__main__":
    fire.Fire(simulate)