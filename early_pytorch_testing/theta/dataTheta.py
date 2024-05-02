import torch
import pickle

class Dataset(torch.utils.data.Dataset): 
    def __init__(self, path: str, nSnps: int, transpose: bool = False):
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.transpose = transpose
        self.nSnps = nSnps
        self.nDatasets = data["nDatasets"]
        self.nSamples = data["nSamples"]
        self.popSizes = torch.from_numpy(data["popSizes"]).float()
        self.snpMatrices = data["charMatrices"]
    
    def __len__(self):
        return self.nDatasets 

    def __getitem__(self, index):
        snps = self.snpMatrices[index][:, :self.nSnps]
        if self.transpose:  
            snps = snps.transpose()
        snps = torch.from_numpy(snps).float() 
        y = self.popSizes[index] 
        return snps.unsqueeze(0), y.unsqueeze(0)
