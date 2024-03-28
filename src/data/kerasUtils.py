import numpy as np
from typing import Callable, Optional, List
from tskit import TreeSequence, Variant

def _sort(mat: np.ndarray, sorting: Callable, axis: int = 0) -> np.ndarray:
    # TODO: Make this capable of doing more complex sorting such as with kmeans
    match axis:
        case 0: slices = [np.s_[i,:] for i in range(mat.shape[axis])]
        case 1: slices = [np.s_[:,i] for i in range(mat.shape[axis])]
        case _: raise ValueError(f"Invalid argument for axis: {axis}")
    distances = [sorting(mat[slices[0]], mat[i]) for i in slices]
    idx = [slice(None)] * mat.ndim
    idx[axis] = np.argsort(distances) # Ignore pycharm warning
    return mat[tuple(idx)]

def _genotypeMatrix(ts: TreeSequence, samples: List[int], nSnps: int, transpose: bool = False, 
        sorting: Optional[Callable] = None) -> np.ndarray:
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
    if sorting:
        _sort(mat, sorting, transpose) 
    return mat

def genotypeMatrix(ts: TreeSequence, nSnps: int, transpose: bool = False, 
        sorting: Optional[Callable] = None, split: bool = False,
        channelsLast: bool = True) -> np.ndarray:
    if split:
        matrices = []
        matrices.append(_genotypeMatrix(ts, ts.samples(1), nSnps, transpose, sorting))
        matrices.append(_genotypeMatrix(ts, ts.samples(1), nSnps, transpose, sorting))
        if channelsLast:
            return np.stack(matrices, axis=-1)
        else:
            return np.array(matrices)
    else:
        return _genotypeMatrix(ts, ts.samples(), nSnps, transpose, sorting)