import os.path as path
from tqdm import tqdm
from pydantic import BaseModel
from typing import List, TypeVar, Optional, Generic
import oyaml as yaml
from pydantic.dataclasses import dataclass
import json
from tskit import TreeSequence, load 
from pydantic import BaseModel, field_serializer, field_validator, ConfigDict, validator, field_validator
import tempfile
import torch
from typing_extensions import Annotated


# class SampleSet(BaseModel):
#     all: float
#     pop1: float
#     pop2: float

# class SummaryStatistics(BaseModel):
#         dxy: float
#         fst: float
#         pi: SampleSet 
#         tajimasD: SampleSet 
#         segregatingSites: SampleSet 
#         # sfs: SampleSet
#         nSnps: int 


U = TypeVar('U') 
class Scenario(BaseModel, Generic[U]): 
    model_config = ConfigDict(arbitrary_types_allowed = True)
    treeSequence: TreeSequence 
    data: U
    # summaryStatistics: SummaryStatistics

    @field_serializer("treeSequence")
    @classmethod
    def serialize_treeSequence(cls, ts, _info):
        with tempfile.SpooledTemporaryFile() as h:
            ts.dump(h)
            h.seek(0)
            return h.read().hex()

    @field_validator("treeSequence", mode="before")
    @classmethod
    def deserialize_treeSequence(cls, v):
        if isinstance(v, str):
            with tempfile.SpooledTemporaryFile() as h:
                h.write(bytes.fromhex(v))
                h.seek(0)
                return load(h)   
        elif isinstance(v, TreeSequence):
            return v


T = TypeVar('T') 
class Simulations(BaseModel, Generic[T]):
    config: T 
    simulations: List[Scenario]
    seed: int

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, ix):
        return self.simulations[ix]
    
    def __iter__(self):
        yield from self.simulations


class Simulator():
    def __init__(self, scenarioType: type, configPath: str, outPrefix: str, 
            nDatasets: int, seed: Optional[int] = None, force: bool = False) -> None:
        """
        Run simulations for a given scenario
        """
        self.nDatasets = nDatasets

        # Create outpath and check for existing
        outPath = f"{outPrefix}.json"
        if path.exists(outPath): 
            if not force:
                quit(f"Aborted: {outPath} already exists")

#         # Load config and validate
        self.scenario = scenarioType()
        config = yaml.safe_load(open(configPath))
        if config:
            self.config = self.scenario.config(**config)
        else:
            quit("Error: empty config file")

        # Create seeds
        if seed: 
            # self.seed = seed
            torch.manual_seed(seed)
        else:
            seed = torch.initial_seed() 
        self.randomSeeds = torch.randint(0, 2**32, (nDatasets,))

        # Run simulations
        simulations = []
        for ix in tqdm(range(self.nDatasets), desc="Simulating data"):
            ts, data = self.scenario(ix, self) 
            # stats = self.getSummaryStats(ts)
            simulations.append(Scenario(treeSequence=ts, data=data))
            # simulations.append(Scenario(treeSequence=ts, data=data, 
                    # summaryStatistics=stats))

        # Output to file 
        print(f"Writing output ...")
        data = Simulations(config=self.config, simulations=simulations, seed=seed)
        with open(outPath, "w") as fh:
            fh.write(data.model_dump_json())
        print("Simulations Complete!")

    # def getSummaryStats(self, ts):
    #     def toSampleSet(values):
    #         return SampleSet(all=values[0], pop1=values[1], pop2=values[2])
    #     sets = [ts.samples(), ts.samples(1), ts.samples(2)]
    #     pair = [ts.samples(1), ts.samples(2)]
    #     stats = SummaryStatistics(
    #         dxy = ts.divergence(pair),
    #         fst = ts.Fst(pair),
    #         pi = toSampleSet(ts.diversity(sets)),
    #         tajimasD = toSampleSet(ts.Tajimas_D(sets)),
    #         segregatingSites = toSampleSet(ts.segregating_sites(sets)),
    #         # sfs = toSampleSet(ts.allele_frequency_spectrum(sets)),
    #     )
    #     return stats