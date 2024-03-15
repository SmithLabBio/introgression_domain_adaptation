#!/usr/bin/env python

import fire
from pydantic import BaseModel, field_validator, model_validator
from typing import Tuple
import torch
from torch.distributions.uniform import Uniform
import tskit
import msprime as mp

# Temporary hack to allow this file to be used in imports and also as an executable
if __name__ == '__main__':
    from simulation import Simulator
else:
    from .simulation import Simulator

class GhostConfig(BaseModel):
    nSamples: int
    sequenceLength: int
    recombinationRate: float
    mutationRate: float
    migrationRateRange: Tuple[float, float]
    popSizeRange: Tuple[int, int]
    divTimeRange: Tuple[int, int]
    ghostDivTimeRange: Tuple[int, int]

    @model_validator(mode="after") # Ignore pycharm warning
    def validate_divtimes(self):
        if not self.divTimeRange[1] <= self.ghostDivTimeRange[0]:
            raise ValueError("divTimeRange must be less than ghostDivTimeRange")

class GhostData(BaseModel):
    popSize: int
    divTime: int
    ghostDivTime: int
    migrationState: int
    migrationRate: float

class Ghost():
    def __init__(self):
        self.config = GhostConfig 

    def __call__(self, ix: int, simulator: Simulator) -> Tuple[tskit.TreeSequence, GhostData]:
        torch.manual_seed(simulator.randomSeeds[ix])
        config = simulator.config
        sizeRange = config.popSizeRange 
        popSize = int(torch.randint(sizeRange[0], sizeRange[1], (1,)).item())
        timeRange = config.divTimeRange
        divTime = int(torch.randint(timeRange[0], timeRange[1], (1,)).item())
        ghostTimeRange = config.ghostDivTimeRange
        ghostDivTime = int(torch.randint(ghostTimeRange[0], ghostTimeRange[1], (1,)).item())
        dem = mp.Demography()
        dem.add_population(name="a", initial_size=popSize)
        dem.add_population(name="b", initial_size=popSize)
        dem.add_population(name="c", initial_size=popSize)
        dem.add_population(name="d", initial_size=popSize)
        dem.add_population(name="e", initial_size=popSize)
        dem.add_population_split(time=ghostDivTime, derived=["b", "c"], ancestral="a")
        dem.add_population_split(time=divTime, derived=["d", "e"], ancestral="c")
        half = simulator.nDatasets // 2    
        if ix > half: 
            migRange = config.migrationRateRange
            migrationRate = Uniform(migRange[0], migRange[1]).sample().item()
            dem.add_symmetric_migration_rate_change(populations=["e", "b"], 
                                                    rate=migrationRate, time=0)
            dem.add_symmetric_migration_rate_change(populations=["e", "b"], 
                                                    rate=0, time=divTime/2)
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
        data = GhostData(popSize=popSize, divTime=divTime, 
                migrationState=migrationState, migrationRate=migrationRate,
                ghostDivTime=ghostDivTime)
        return mts, data 

def run(configPath, outPrefix, nDatasets, force=False):
    s = Simulator(scenarioType=Ghost, configPath=configPath,
            outPrefix=outPrefix, nDatasets=nDatasets, force=force)

if __name__ == "__main__":
    fire.Fire(run)