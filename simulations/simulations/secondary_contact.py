#!/usr/bin/env python

import fire
from typing import Tuple
import tskit
import msprime as mp
import numpy as np
from .simulator import Scenario, ScenarioData, run_simulations


class SecondaryContactData(ScenarioData):
    population_size: int
    divergence_time: int
    migration_state: int
    migration_rate: float

class SecondaryContact(Scenario):
    _data_class = SecondaryContactData
    n_samples: int
    sequence_length: int
    recombination_rate: float
    mutation_rate: float
    migration_rate_range: Tuple[float, float]
    population_size_range: Tuple[int, int]
    divergence_time_range: Tuple[int, int]

    def __call__(self, ix: int, seed: int, n_datasets: int) -> Tuple[tskit.TreeSequence, SecondaryContactData]:
        rng = np.random.default_rng(seed)
        population_size = rng.integers(self.population_size_range[0], self.population_size_range[1])
        divergence_time = rng.integers(self.divergence_time_range[0], self.divergence_time_range[1])
        dem = mp.Demography()
        dem.add_population(name="c", initial_size=population_size)
        dem.add_population(name="d", initial_size=population_size)
        dem.add_population(name="e", initial_size=population_size)
        dem.add_population_split(time=divergence_time, derived=["d", "e"], ancestral="c")
        half = n_datasets // 2    
        if ix >= half: 
            migration_rate = rng.uniform(self.migration_rate_range[0], self.migration_rate_range[1])
            dem.add_symmetric_migration_rate_change(populations=["d", "e"], 
                    time=0, rate=migration_rate/population_size)
            dem.add_symmetric_migration_rate_change(populations=["d", "e"], 
                    time=divergence_time//2, rate=0)
            migration_state = 1
        else:
            migration_rate = 0
            migration_state = 0
        dem.sort_events()
        ts = mp.sim_ancestry(
            samples=dict(d=self.n_samples, e=self.n_samples),
            demography=dem, 
            random_seed=rng.integers(0, 2**32), 
            sequence_length=self.sequence_length, 
            recombination_rate=self.recombination_rate)
        mts = mp.sim_mutations(ts, rate=self.mutation_rate, random_seed=rng.integers(0, 2**32))
        data = SecondaryContactData(
            population_size=population_size, 
            divergence_time=divergence_time, 
            migration_state=migration_state, 
            migration_rate=migration_rate)

        return mts, data 

def run(config_path, out_path, n_datasets, force=False):
    run_simulations(scenario_type=SecondaryContact, config_path=config_path,
            out_path=out_path, n_datasets=n_datasets, force=force)

def cli():
    fire.Fire(run)

if __name__ == "__main__":
    fire.Fire(run)