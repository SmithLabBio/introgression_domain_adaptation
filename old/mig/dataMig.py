import torch
import pickle
import numpy as np

class Dataset(torch.utils.data.Dataset): 
    def __init__(self, path: str, nSnps: int):
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.nSnps = nSnps
        self.nDatasets = data["nDatasets"]
        self.nSamples = data["nSamples"]
        self.migrationStates = torch.from_numpy(data["migrationStates"]).long()
        fullSnpMatrices = data["charMatrices"]
        snpMatrices = []
        for i in fullSnpMatrices:
            snpMatrices.append(i[:, :nSnps])
        self.snpMatrices = torch.from_numpy(np.array(snpMatrices)).float()
    
    def __len__(self):
        return self.nDatasets 

    def __getitem__(self, index):
        snps = self.snpMatrices[index].unsqueeze(0)
        y = self.migrationStates[index] 
        return snps, y
