import torch
import pickle
import numpy as np

def positionsToDistances(pos):
    dist = torch.empty_like(pos)
    dist[1:] = pos[1:] - pos[:-1]
    dist[0] = pos[0]
    return dist

class Dataset(torch.utils.data.Dataset): 
    def __init__(self, path: str, nSnps: int):
        data = np.load(path, allow_pickle=True)
        self.nSnps = nSnps
        self.config = pickle.loads(data["config"])
        self.nDatasets = self.config["nDatasets"]
        self.positions = data["positions"]
        self.charMatrices = data["charMatrices"]
        self.migrationRates = torch.from_numpy(data["migrationRates"]).float()
    
    def __len__(self):
        return self.nDatasets 

    def __getitem__(self, index):
        charMatrix = torch.from_numpy(self.charMatrices[index][:self.nSnps, :]) 
        positions = torch.from_numpy(self.positions[index][:self.nSnps])
        distances = positionsToDistances(positions).unsqueeze(0)
        x = torch.cat((distances, charMatrix), 0).float() 
        y = self.migrationRates[index] 
        return x, y 
