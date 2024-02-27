import torch
import pickle
import tskit
from typing import List, TypeVar
from sim import Simulations
from secondaryContact import SecondaryContactConfig, SecondaryContactData
import json

# def positionsToDistances(pos):
#     dist = torch.empty_like(pos)
#     dist[1:] = pos[1:] - pos[:-1]
#     dist[0] = pos[0]
#     return dist

def getGenotypeMatrix(ts, samples: List[int] , nSnps: int):
    """Construct tensor from treesequence"""
    # TODO: make this output matrices with the dimension for channels
    var = tskit.Variant(ts, samples=samples) 
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
        self.simulations = Simulations[SecondaryContactConfig].model_validate_json(jsonData)
        self.snpMatrices = []
        for i in self.simulations:
            ts = self.simulations[i].treeSequence
            if split:
                popMatrices = []
                popMatrices.append(getGenotypeMatrix(ts, ts.samples(1), 500))
                popMatrices.append(getGenotypeMatrix(ts, ts.samples(2), 500))
                self.snpMatrices.append(popMatrices)
            else:
                self.snpMatrices.append(getGenotypeMatrix(ts, ts.samples(), 500))
    
    def __len__(self):
        return len(self.simulations) 

    def __getitem__(self, index):
        if self.test:
            return self.snpMatrices[index]
        else:
            return self.snpMatrices[index], self.migrationStates[index]
        snps = self.snpMatrices[index][:, :self.nSnps]
        if self.transpose:  
            snps = snps.transpose()
        snps = torch.from_numpy(snps).float() 
        pos = torch.from_numpy(self.positions[index][:self.nSnps]).float()
        if self.toDistance:
            pos = positionsToDistances(pos)
        y = self.migrationRates[index] 
        return snps, pos, y


d = Dataset("secondaryContact1/secondaryContact1-1.json", 500)
