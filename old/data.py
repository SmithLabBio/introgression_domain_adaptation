import torch
import pickle
import numpy as np

def positionsToDistances(pos):
    dist = torch.empty_like(pos)
    dist[1:] = pos[1:] - pos[:-1]
    dist[0] = pos[0]
    return dist

# class Dataset(torch.utils.data.Dataset): 
#     def __init__(self, path: str, nSnps: int):
#         data = np.load(path, allow_pickle=True)
#         self.nSnps = nSnps
#         self.config = pickle.loads(data["config"])
#         self.summaryStates = data["summaryStats"]
#         self.nDatasets = self.config["nDatasets"]
#         self.positions = data["positions"]
#         self.charMatrices = data["charMatrices"]
#         self.migrationRates = torch.from_numpy(data["migrationRates"]).float()
    
#     def __len__(self):
#         return self.nDatasets 

#     def __getitem__(self, index):
#         charMatrix = torch.from_numpy(self.charMatrices[index][:, :self.nSnps]) 
#         positions = torch.from_numpy(self.positions[index][:self.nSnps])
#         distances = positionsToDistances(positions).unsqueeze(0)
#         x = torch.cat((distances, charMatrix), 0).float() 
#         y = self.migrationRates[index] 
#         return x, y 

class Dataset(torch.utils.data.Dataset): 
    def __init__(self, path: str, nSnps: int, transpose: bool = False, 
                 toDistance: bool = True):
        data = np.load(path, allow_pickle=True)
        self.transpose = transpose
        self.nSnps = nSnps
        self.toDistance = toDistance
        self.config = pickle.loads(data["config"])
        self.summaryStates = pickle.loads(data["summaryStats"])
        self.nDatasets = self.config["nDatasets"]
        self.positions = data["positions"]
        self.snpMatrices = data["charMatrices"]
        self.migrationRates = torch.from_numpy(data["migrationRates"]).float()
    
    def __len__(self):
        return self.nDatasets 

    def __getitem__(self, index):
        snps = self.snpMatrices[index][:, :self.nSnps]
        if self.transpose:  
            snps = snps.transpose()
        snps = torch.from_numpy(snps).float() 
        pos = torch.from_numpy(self.positions[index][:self.nSnps]).float()
        if self.toDistance:
            pos = positionsToDistances(pos)
        y = self.migrationRates[index] 
        return snps, pos, y