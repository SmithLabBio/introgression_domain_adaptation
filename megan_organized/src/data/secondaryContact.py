#!/usr/bin/env python

import fire
from pydantic import BaseModel
from typing import Tuple
import torch
from torch.distributions.uniform import Uniform
import tskit
import msprime as mp

# Temporary hack to allow this file to be used in imports and also as an executable
if __name__ == '__main__':
    from simulation import Simulator
else:
    from simulation import Simulator


class SecondaryContactConfig(BaseModel):
    nSamples: int
    sequenceLength: int
    recombinationRate: float
    mutationRate: float
    migrationRateRange: Tuple[float, float]
    popSizeRange: Tuple[int, int]
    divTimeRange: Tuple[int, int]

class SecondaryContactData(BaseModel):
    popSize: int
    divTime: int
    migrationState: int
    migrationRate: float

class SecondaryContact():
    def __init__(self):
        self.config = SecondaryContactConfig 

    def __call__(self, ix: int, simulator: Simulator) -> Tuple[tskit.TreeSequence, SecondaryContactData]:
        torch.manual_seed(simulator.randomSeeds[ix])
        config = simulator.config
        sizeRange = config.popSizeRange 
        popSize = int(torch.randint(sizeRange[0], sizeRange[1], (1,)).item())
        timeRange = config.divTimeRange
        divTime = int(torch.randint(timeRange[0], timeRange[1], (1,)).item())
        dem = mp.Demography()
        dem.add_population(name="c", initial_size=popSize)
        dem.add_population(name="d", initial_size=popSize)
        dem.add_population(name="e", initial_size=popSize)
        dem.add_population_split(time=divTime, derived=["d", "e"], ancestral="c")
        half = simulator.nDatasets // 2    
        if ix > half: 
            migRange = config.migrationRateRange
            migrationRate = Uniform(migRange[0], migRange[1]).sample().item()
            dem.add_symmetric_migration_rate_change(populations=["d", "e"], 
                    time=0, rate=migrationRate)
            dem.add_symmetric_migration_rate_change(populations=["d", "e"], 
                    time=divTime//2, rate=0)
            migrationState = 1
        else:
            migrationRate = 0
            migrationState = 0
        dem.sort_events()
        ts = mp.sim_ancestry(samples={"d": config.nSamples, "e": 
                config.nSamples}, demography=dem, 
                random_seed=torch.randint(0, 2**32, (1,)).item(), 
                sequence_length=config.sequenceLength, 
                recombination_rate=config.recombinationRate)
        mts = mp.sim_mutations(ts, rate=config.mutationRate, 
                random_seed=torch.randint(0, 2**32, (1,)).item())
        data = SecondaryContactData(popSize=popSize, divTime=divTime, 
                migrationState=migrationState, migrationRate=migrationRate)
        return mts, data 

def run(configPath, outPrefix, nDatasets, force=False):
    s = Simulator(scenarioType=SecondaryContact, configPath=configPath,
            outPrefix=outPrefix, nDatasets=nDatasets, force=force)

if __name__ == "__main__":
    fire.Fire(run)