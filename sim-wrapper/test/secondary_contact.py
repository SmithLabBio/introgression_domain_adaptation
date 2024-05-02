#!/usr/bin/env python

import fire
from typing import Tuple
import numpy as np
import msprime as mp
from sim_wrapper.simulator import Scenario, ScenarioData, run_simulations

class SecondaryContactData(ScenarioData):
    popSize: int
    divTime: int
    migrationState: int
    migrationRate: float

class SecondaryContact(Scenario):
    _data_class = SecondaryContactData
    nSamples: int
    sequenceLength: int
    recombinationRate: float
    mutationRate: float
    migrationRateRange: Tuple[float, float]
    popSizeRange: Tuple[int, int]
    divTimeRange: Tuple[int, int]

    def __call__(self, ix, seed, n_datasets):
        rng = np.random.default_rng(seed)
        sizeRange = self.popSizeRange 
        popSize = rng.integers(sizeRange[0], sizeRange[1])
        timeRange = self.divTimeRange
        divTime = rng.integers(timeRange[0], timeRange[1])
        dem = mp.Demography()
        dem.add_population(name="c", initial_size=popSize)
        dem.add_population(name="d", initial_size=popSize)
        dem.add_population(name="e", initial_size=popSize)
        dem.add_population_split(time=divTime, derived=["d", "e"], ancestral="c")
        half = n_datasets // 2    
        if ix >= half: 
            migRange = self.migrationRateRange
            migrationRate = rng.uniform(migRange[0], migRange[1])
            dem.add_symmetric_migration_rate_change(populations=["d", "e"], 
                    time=0, rate=migrationRate)
            dem.add_symmetric_migration_rate_change(populations=["d", "e"], 
                    time=divTime//2, rate=0)
            migrationState = 1
        else:
            migrationRate = 0
            migrationState = 0
        dem.sort_events()
        ts = mp.sim_ancestry(samples=dict(d=self.nSamples, e=self.nSamples),
                demography=dem, random_seed=rng.integers(0, 2**32), 
                sequence_length=self.sequenceLength, 
                recombination_rate=self.recombinationRate)
        mts = mp.sim_mutations(ts, rate=self.mutationRate, 
                random_seed=rng.integers(0, 2**32))
        data = SecondaryContactData(popSize=popSize, divTime=divTime, 
                migrationState=migrationState, migrationRate=migrationRate)
        return mts, data 

def run(config_path, out_prefix, n_datasets, force=False):
    run_simulations(scenario_type=SecondaryContact, config_path=config_path, out_prefix=out_prefix, 
                    n_datasets=n_datasets, force=force)

if __name__ == "__main__":
    fire.Fire(run)