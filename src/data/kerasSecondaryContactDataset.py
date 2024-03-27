from tensorflow import keras
from tensorflow.keras.utils import to_categorical # Ignore pycharm
import numpy as np
from typing import Optional, Callable

from simulation import Simulations
from secondaryContact import SecondaryContactConfig
from kerasUtils import genotypeMatrix

class Dataset():
    def __init__(self, path: str, nSnps: int, transpose: bool = False,
            sorting: Optional[Callable] = None, split: bool = False,
            categorical: bool = False):
        with open(path, "rb") as fh:
            jsonData = fh.read()
        self.simulations = Simulations[SecondaryContactConfig].model_validate_json(jsonData)
        snpMatrices = []
        migrationStates = []
        for ix, s in enumerate(self.simulations):
            ts = s.treeSequence 
            snpMatrices.append(genotypeMatrix(ts, nSnps, transpose, sorting, split))
            migrationStates.append(self.simulations[ix].data["migrationState"])
        self.snps = np.array(snpMatrices)
        if categorical:
            self.migrationStates = to_categorical(migrationStates, num_classes=2)
        else: 
            self.migrationStates = np.array(migrationStates)
        self.shape = self.snps.shape[1:]