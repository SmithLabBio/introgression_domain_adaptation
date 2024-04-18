from dataclasses import dataclass
import torch
import json
from tskit import TreeSequence, Variant
from typing import List, TypeVar, Optional
from lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from .simulation import Simulations
from .secondaryContact import SecondaryContactConfig, SecondaryContactData

@dataclass
class Data:
    snps: torch.Tensor
    distances: torch.Tensor

def positionsToDistances(pos: torch.Tensor) -> torch.Tensor:
    dist = torch.empty_like(pos)
    dist[1:] = pos[1:] - pos[:-1]
    dist[0] = pos[0]
    return dist

def getGenotypeMatrix(ts: TreeSequence, samples: List[int] , nSnps: int) -> torch.Tensor:
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
        self.simulations = Simulations[SecondaryContactConfig].model_validate_json(jsonData)
        self.snpMatrices = []
        self.distanceArrays = []
        for i in self.simulations:
            ts = i.treeSequence 
            if split:
                n_samples = len(ts.samples())
                pop1 = list(range(0, n_samples//2))
                pop2 = list(range(n_samples//2, n_samples))
                popMatrices = []
                popMatrices.append(getGenotypeMatrix(ts, pop1, nSnps))
                popMatrices.append(getGenotypeMatrix(ts, pop2, nSnps))
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
        migrationState = self.simulations[ix].data["migrationState"]
        return Data(snps, distances), migrationState 


# class AdaptDataset(torch.utils.data.Dataset):
#     def __init__(self, srcTrain: Dataset, targetTrain: Dataset, 
#             srcValidation: Optional[Dataset] = None, 
#             targetValidation: Optional[Dataset] = None):
