import pickle
import os.path as path
import msprime as mp
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel
from typing import Optional
import oyaml as yaml

# TODO: Replace dictionaries with dataclass object or pydantic

class Simulation():
    def __init__(self, scenarioType, configPath: str, outPrefix: str, 
            nDatasets: int, seed: Optional[int] = None, force: bool = False):
        outPath = f"{outPrefix}.pickle"
        if path.exists(outPath): 
            if not force:
                quit(f"Aborted: {outPath} already exists")
        self.simType = scenarioType()
        config = yaml.safe_load(open(configPath))
        if config:
            self.config = self.simType.configType(**config)
        else:
            quit("Error: empty config file")
        if seed: 
            self.seed = seed
        else:
            self.seed = int(np.random.rand() * (2**32 - 1))
        self.rng = np.random.default_rng(seed=self.seed)
        self.nDatasets = nDatasets
        self.ancestrySeeds = self.rng.integers(2**32, size=nDatasets)
        self.mutationSeeds = self.rng.integers(2**32, size=nDatasets)
        self.ix = 0 

        simData = []
        for i in tqdm(range(self.nDatasets), desc="Simulating data"):
            data = self.simType.sim(self)
            self.getSummaryStats(data)
            simData.append(data)
            self.ix += 1

        print(f"Writing output ...")
        data = dict(**self.config.model_dump(), simulations=simData)
        with open(outPath, "wb") as fh:
            pickle.dump(data, fh)
        print("Simulations Complete!")


    def getSummaryStats(self, data):
        def toSampleDict(values):
            return dict(all=values[0], pop1=values[1], pop2=values[2])
        ts = data["treeSequence"]
        data["dxy"] = ts.divergence(sample_sets=[ts.samples(1), ts.samples(2)])
        data["fst"] = ts.Fst(sample_sets=[ts.samples(1), ts.samples(2)])
        data["pi"] = toSampleDict(ts.diversity(
                sample_sets=[ts.samples(), ts.samples(1), ts.samples(2)]))
        data["tajimasD"] = toSampleDict(ts.Tajimas_D(
                sample_sets=[ts.samples(), ts.samples(1), ts.samples(2)]))
        data["segregatingSites"] = toSampleDict(ts.segregating_sites(
                sample_sets=[ts.samples(), ts.samples(1), ts.samples(2)]))
        data["sfs"] = toSampleDict(ts.allele_frequency_spectrum(
                sample_sets=[ts.samples(), ts.samples(1), ts.samples(2)]))
        data["nSnps"] = len(ts.sites())