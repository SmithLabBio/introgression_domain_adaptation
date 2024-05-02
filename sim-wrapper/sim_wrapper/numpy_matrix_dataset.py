from typing import Optional, Callable
from numpy import array
from genotype_matrix import genotype_matrix 


from .simulator import Simulations, Replicate

class Dataset():
    def __init__(self, scenario: type, path: str, nSnps: int, transpose: bool = False,
            sorting: Optional[Callable] = None, split: bool = False):
        with open(path, "rb") as fh:
            jsonData = fh.read()
        simulations = Simulations[scenario, scenario._data_class].model_validate_json(jsonData)
        assert len(simulations.simulations) > 0
        for i in simulations.simulations[0].data.model_fields_set:
            l = []
            for rep in simulations.simulations: 
                l.append(getattr(rep.data, i))
            setattr(self, i, array(l))
        
        #TODO: Decide how to create snp matrices

        # snpMatrices = []
        # migrationStates = []
        # for ix, s in enumerate(self.simulations):
            # ts = s.treeSequence 
            # snpMatrices.append(genotypeMatrix(ts, nSnps, transpose, sorting, split))
            # migrationStates.append(self.simulations[ix].data["migrationState"])
        # self.snps = np.array(snpMatrices)
        # if categorical:
        #     self.migrationStates = to_categorical(migrationStates, num_classes=2)
        # else: 
        #     self.migrationStates = np.array(migrationStates)
        # self.shape = self.snps.shape[1:]