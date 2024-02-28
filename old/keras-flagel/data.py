import tskit
import numpy as np
import pickle

def getGenotypeMatrix(ts, samples, nSnps: int, transpose: bool = False):
    """Construct numpy matrix from treesequence"""
    # TODO: make this output matrices with the dimension for channels
    var = tskit.Variant(ts, samples=samples) 
    if transpose:
        shape = (nSnps, len(samples))
    else:
        shape = (len(samples), nSnps)
    mat = np.zeros(shape=shape, dtype=np.int32)
    for site in range(nSnps):
        var.decode(site)
        if transpose:
            mat[site, :] = var.genotypes
        else:
            mat[:, site] = var.genotypes
    return mat

def getData(path: str, nSnps: int, transpose: bool =False, split: bool = False):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    snpMatrices = []
    migrationStates = []
    for i in data["simulations"]:
        ts = i["treeSequence"]
        if split:
            popMatrices = []
            popMatrices.append(getGenotypeMatrix(ts, ts.samples(1), 500, transpose=transpose))
            popMatrices.append(getGenotypeMatrix(ts, ts.samples(2), 500, transpose=transpose))
            snpMatrices.append(np.array(popMatrices))
        else:
            snpMatrices.append(getGenotypeMatrix(ts, ts.samples(), 500, transpose=transpose))
        migrationStates.append(i["migrationState"])
    snpMatrices = np.array(snpMatrices)
    migrationStates = np.array(migrationStates)
    return data, snpMatrices, migrationStates