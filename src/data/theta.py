#!/usr/bin/env python

import fire
from pydantic import BaseModel
from typing import Tuple
import torch
from torch.distributions.uniform import Uniform
import tskit
import msprime as mp
from .simulation import Simulator


class ThetaConfig(BaseModel):
    nSamples: int
    mutationRate: float
    sequenceLength: int
    recombinationRate: float
    popSizeRange: Tuple[int, int]

class ThetaData(BaseModel):
    populationSize: int

class Theta():
    def __init__(self):
        self.config = ThetaConfig 

    def __call__(self, ix: int, simulator: Simulator) -> Tuple[tskit.TreeSequence, ThetaData]:
        torch.manual_seed(simulator.randomSeeds[ix])
        config = simulator.config
        sizeRange = config.popSizeRange 
        popSize = int(torch.randint(sizeRange[0], sizeRange[1], (1,)).item())
        ts = mp.sim_ancestry(samples=config.nSamples, 
                random_seed=torch.randint(0, 2**32, (1,)).item(), 
                sequence_length=config.sequenceLength, 
                recombination_rate=config.recombinationRate)
        mts = mp.sim_mutations(ts, rate=config.mutationRate, 
                random_seed=torch.randint(0, 2**32, (1,)).item())
        data = ThetaData(populationSize=popSize)
        return mts, data 

def run(configPath, outPrefix, nDatasets, force=False):
    s = Simulator(scenarioType=Theta, configPath=configPath,
            outPrefix=outPrefix, nDatasets=nDatasets, force=force)

if __name__ == "__main__":
    fire.Fire(run)