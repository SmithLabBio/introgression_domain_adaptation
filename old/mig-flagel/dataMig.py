import torch
import pickle

class Dataset(torch.utils.data.Dataset): 
    def __init__(self, path: str, nSnps: int):
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.nSnps = nSnps
        self.nDatasets = data["nDatasets"]
        self.nSamples = data["nSamples"]
        self.migrationStates = torch.from_numpy(data["migrationStates"]).long()
        self.snpMatrices = data["charMatrices"]
    
    def __len__(self):
        return self.nDatasets 

    def __getitem__(self, index):
        snps = self.snpMatrices[index][:, :self.nSnps]
        snps = torch.from_numpy(snps).float() 
        y = self.migrationStates[index] 
        return snps.unsqueeze(0), y
