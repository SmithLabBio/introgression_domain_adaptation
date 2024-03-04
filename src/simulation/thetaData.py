import torch
from tskit import TreeSequence, Variant
from typing import List, TypeVar
from .sim import Simulations
from .theta import ThetaConfig, ThetaData
import json

def positionsToDistances(pos):
    dist = torch.empty_like(pos)
    dist[1:] = pos[1:] - pos[:-1]
    dist[0] = pos[0]
    return dist

def getGenotypeMatrix(ts: TreeSequence, samples: List[int] , nSnps: int):
    """Construct tensor from treesequence"""
    # TODO: make this output matrices with the dimension for channels
    var = Variant(ts, samples=samples) 
    mat = torch.empty(size=(len(samples), nSnps), dtype=torch.float)
    for site in range(nSnps):
        var.decode(site)
        mat[:, site] = torch.tensor(var.genotypes)
    return mat

class Dataset(torch.utils.data.Dataset): 
    def __init__(self, path: str, nSnps: int, split: bool = True):
        with open(path, "rb") as fh:
            jsonData = fh.read()
        # TODO: Fully parse json into nested models instead of into dictionaries
        self.simulations = Simulations[ThetaConfig].model_validate_json(jsonData)
        self.snpMatrices = []
        self.distanceArrays = []
        for i in self.simulations:
            ts = i.treeSequence 
            if split:
                popMatrices = []
                popMatrices.append(getGenotypeMatrix(ts, ts.samples(1), nSnps))
                popMatrices.append(getGenotypeMatrix(ts, ts.samples(2), nSnps))
                self.snpMatrices.append(popMatrices)
            else:
                self.snpMatrices.append(getGenotypeMatrix(ts, ts.samples(), nSnps))
            distances = torch.tensor(ts.tables.sites.position[:nSnps], dtype=torch.float)
            self.distanceArrays.append(distances)

    def __len__(self):
        return len(self.simulations) 

    def __getitem__(self, ix):
        snps = self.snpMatrices[ix]
        distances = self.distanceArrays[ix]
        theta = torch.tensor(self.simulations[ix].data["populationSize"], dtype=torch.float).unsqueeze(0)
        return snps, distances, theta 
