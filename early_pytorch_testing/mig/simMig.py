#!/usr/bin/env python3

import fire
import pickle
import os.path as path
import msprime as mp
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel
from typing import Tuple
import oyaml as yaml


class Config(BaseModel):
    seed:int
    nDatasets: int
    nSamples: int
    sequenceLength: int
    recombinationRate: float
    mutationRate: float
    migrationRateRange: Tuple[float, float]
    popSizeRange: Tuple[int, int]
    divTimeRange: Tuple[int, int]

def simulate(configPath, outPrefix, nDatasets, seed=None, force=False) -> None: 
    outPath = f"{outPrefix}.pickle"
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
    ancestrySeeds = np.random.randint(2**32, size=nDatasets)
    mutationSeeds = np.random.randint(2**32, size=nDatasets)

    # Div times and pop sizes
    popSizes = np.random.randint(config.popSizeRange[0], config.popSizeRange[1], 
            size=nDatasets)
    divTimes = np.random.randint(config.divTimeRange[0], config.divTimeRange[1],
            size=nDatasets)

    # Migration state and rate
    half = nDatasets // 2 
    otherHalf = nDatasets - half
    migrationStates = np.concatenate((np.zeros(half), np.ones(otherHalf)))
    migrationRates = np.concatenate((
        migrationStates[:half], 
        np.random.uniform(config.migrationRateRange[0], 
                          config.migrationRateRange[1], size=otherHalf)))

    # Initialize containers  
    charMatrices = []
    positions = []
    statDict = dict(all=[], pop1=[], pop2=[])
    summaryStats = dict(dxy=[], fst=[], 
            pi=statDict.copy(), 
            tajimasD=statDict.copy(), 
            segregatingSites=statDict.copy(),
            sfs=statDict.copy()) 

    # Simulate
    for i in tqdm(range(nDatasets), desc="Simulating data"):
        dem = mp.Demography()
        dem.add_population(name="a", initial_size=popSizes[i])
        dem.add_population(name="b", initial_size=popSizes[i])
        dem.add_population(name="c", initial_size=popSizes[i])
        dem.add_population_split(time=divTimes[i], derived=["b", "c"], ancestral="a")
        if migrationStates[i] == 1:
            dem.add_symmetric_migration_rate_change(populations=["b", "c"], 
                    rate=migrationRates[i], time=0)
            dem.add_symmetric_migration_rate_change(populations=["b", "c"], 
                    rate=0, time=divTimes[i]/2)
            dem.sort_events()
        ts = mp.sim_ancestry(samples={"b": config.nSamples, "c": config.nSamples},
                demography=dem, random_seed=ancestrySeeds[i], 
                sequence_length=config.sequenceLength, 
                recombination_rate=config.recombinationRate)
        mts = mp.sim_mutations(ts, rate=config.mutationRate, 
                random_seed=mutationSeeds[i])
        positions.append(mts.tables.sites.position)
        snps = mts.genotype_matrix().transpose() # Transpose to put haplotypes first
        charMatrices.append(snps)  

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

    print(f"Writing output ...")
    data = dict(**config.model_dump(), popSizes=popSizes, divTimes=divTimes, 
                migrationRates=migrationRates, migrationStates=migrationStates,
                charMatrices=charMatrices, summaryStats=summaryStats)
    with open(outPath, "wb") as fh:
        pickle.dump(data, fh)

    print("Simulations Complete!")

if __name__ == "__main__":
    fire.Fire(simulate)