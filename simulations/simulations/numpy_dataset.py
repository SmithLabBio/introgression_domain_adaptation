from typing import Optional, Callable
import numpy as np
from .simulator import Simulations
from .genotype_matrix import genotype_matrix


class NumpySnpDataset():
    def __init__(self, scenario: type, path: str, field: str, n_snps: int, split: bool = False,
            sorting: Optional[Callable] = None, transpose: bool = False):
        with open(path, "r") as fh:
            json_data = fh.read()
        simulations = Simulations[scenario, scenario._data_class].model_validate_json(json_data)
        snps = []
        labels = []
        for rep in simulations:
            snps.append(genotype_matrix(ts=rep.treeSequence, n_snps=n_snps, transpose=transpose, 
                    sorting=sorting, split=split))
            labels.append(getattr(rep.data, field))
        if split:
            self.x = np.array(snps)
        else:
            self.x = np.expand_dims(np.array(snps), -1)
        self.labels = np.array(labels)


class NumpyAfsDataset():
    def __init__(self, scenario: type, path: str, field: str, expand_dims=False, polarized=False, normalized=False):
        with open(path, "r") as fh:
            json_data = fh.read()
        simulations = Simulations[scenario, scenario._data_class].model_validate_json(json_data)
        afs_list = []
        labels = []
        for rep in simulations:
            ts = rep.treeSequence
            n_samples = len(ts.samples())
            pop1 = list(range(0, n_samples//2))
            pop2 = list(range(n_samples//2, n_samples))
            afs = rep.treeSequence.allele_frequency_spectrum(sample_sets=[pop1, pop2], span_normalise=False, polarised=polarized)
            if normalized:
                # TODO: possible to reduce memory usage? 
                sum = np.sum(afs)
                afs = afs / sum 
            afs_list.append(afs)
            labels.append(getattr(rep.data, field))
        if expand_dims:
            self.x = np.expand_dims(np.array(afs_list), -1)
        else:
            self.x = np.array(afs_list)
        self.labels = np.array(labels)

    def save(self, outpath):
        np.savez(outpath, x=self.x, labels=self.labels)

    @classmethod 
    def load(cls, path):
        obj = cls.__new__(cls)
        data = np.load(path)
        obj.x = data["x"]
        obj.labels = data["labels"]
        return obj

    @classmethod
    def from_arrays(cls, x, labels, expand_dims=False):
        obj = cls.__new__(cls)
        if expand_dims:
            obj.x = np.expand_dims(x, -1)
        else:
            obj.x = x 
        obj.labels = labels
        return obj

