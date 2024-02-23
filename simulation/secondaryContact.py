#!/usr/bin/env python

import msprime as mp
from pydantic import BaseModel
from typing import Tuple
import fire
import simulation

# TODO: Replace dictionary with dataclass object or pydantic

class Config(BaseModel):
    nSamples: int
    sequenceLength: int
    recombinationRate: float
    mutationRate: float
    migrationRateRange: Tuple[float, float]
    popSizeRange: Tuple[int, int]
    divTimeRange: Tuple[int, int]

class SecondaryContact():
    def __init__(self):
        self.configType = Config 

    def sim(self, sim):
        sizeRange = sim.config.popSizeRange 
        popSize = sim.rng.integers(sizeRange[0], sizeRange[1])
        timeRange = sim.config.divTimeRange
        divTime = sim.rng.integers(timeRange[0], timeRange[1])
        dem = mp.Demography()
        dem.add_population(name="a", initial_size=popSize)
        dem.add_population(name="b", initial_size=popSize)
        dem.add_population(name="c", initial_size=popSize)
        dem.add_population_split(time=divTime, derived=["b", "c"], ancestral="a")
        half = sim.nDatasets // 2    
        if sim.ix > half: 
            migRange = sim.config.migrationRateRange
            migrationRate = sim.rng.uniform(migRange[0], migRange[1])
            dem.add_symmetric_migration_rate_change(populations=["b", "c"], 
                rate=migrationRate, time=0)
            migrationState = 1
        else:
            migrationRate = 0
            migrationState = 0
        dem.sort_events()
        ts = mp.sim_ancestry(samples={"b": sim.config.nSamples, "c": 
                sim.config.nSamples}, demography=dem, 
                random_seed=sim.ancestrySeeds[sim.ix], 
                sequence_length=sim.config.sequenceLength, 
                recombination_rate=sim.config.recombinationRate)
        mts = mp.sim_mutations(ts, rate=sim.config.mutationRate, 
                random_seed=sim.mutationSeeds[sim.ix])
        data = dict(popSize=popSize, divTime=divTime, 
                migrationState=migrationState, migrationRate=migrationRate,
                treeSequence=mts)
        return data 

def run(configPath, outPrefix, nDatasets, force=False):
    s = simulation.Simulation(scenarioType=SecondaryContact, configPath=configPath, # Ignore pylance error
            outPrefix=outPrefix, nDatasets=nDatasets, force=force)

if __name__ == "__main__":
    fire.Fire(run)