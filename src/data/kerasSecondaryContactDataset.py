from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
from simulation import Simulations
from secondaryContact import SecondaryContactConfig
from tskit import TreeSequence, Variant

def getGenotypeMatrix(ts: TreeSequence, nSnps: int, transpose=False) -> np.ndarray:
    var = Variant(ts, samples=ts.samples())
    if transpose:
        shape = (nSnps, len(ts.samples()))
    else:
        shape = (len(ts.samples()), nSnps)
    mat = np.empty(shape=shape)
    for site in range(nSnps):
        var.decode(site)
        if transpose:
            mat[site, :] = var.genotypes
        else:
            mat[:, site] = var.genotypes
    return mat

class Dataset():
    def __init__(self, path: str, nSnps: int, transpose=False):
        with open(path, "rb") as fh:
            jsonData = fh.read()
        self.simulations = Simulations[SecondaryContactConfig].model_validate_json(jsonData)
        snpMatrices = []
        migrationStates = []
        for ix, s in enumerate(self.simulations):
            ts = s.treeSequence 
            snpMatrices.append(getGenotypeMatrix(ts, nSnps, transpose=transpose))
            migrationStates.append(self.simulations[ix].data["migrationState"])
        self.snps = np.array(snpMatrices)
        # self.migrationStates = np.array(migrationStates)
        self.migrationStates = to_categorical(migrationStates, num_classes=2)
        self.shape = self.snps.shape[1:]