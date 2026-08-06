import os.path as path
from typing import List, TypeVar, Optional, Generic, Any, Tuple
import oyaml as yaml
from pydantic import BaseModel, field_serializer, field_validator, ConfigDict, field_validator
import base64
from tskit import TreeSequence, load 
import tempfile
import random
from tqdm import tqdm


class ScenarioData(BaseModel):
    pass

class Scenario(BaseModel): 
    pass

U = TypeVar('U') 
class Replicate(BaseModel, Generic[U]): 
    model_config = ConfigDict(arbitrary_types_allowed = True)
    data: U
    treeSequence: TreeSequence  #TODO: Change this field name to have underscore to make it consistent.

    @field_serializer("treeSequence")
    @classmethod
    def serialize_treeSequence(cls, ts, _info):
        with tempfile.SpooledTemporaryFile() as h:
            ts.dump(h)
            h.seek(0)
            return base64.b64encode(h.read())

    @field_validator("treeSequence", mode="before")
    @classmethod
    def deserialize_treeSequence(cls, v):
        if isinstance(v, str):
            with tempfile.SpooledTemporaryFile() as h:
                h.write(base64.b64decode(v))
                h.seek(0)
                return load(h)   
        elif isinstance(v, TreeSequence):
            return v

T = TypeVar('T') 
class Simulations(BaseModel, Generic[T, U]):
    scenario: str
    config: T 
    seed: int
    simulations: List[Replicate[U]]

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, ix):
        return self.simulations[ix]
    
    def __iter__(self):
        yield from self.simulations

def run_simulations(scenario_type: type, config_path: str, out_path: str, n_datasets: int, 
                    seed: Optional[int] = None, force: bool = False) -> None:

    # Create outpath and check for existing
    if path.exists(out_path): 
        if not force:
            quit(f"Aborted: {out_path} already exists")

    # Load and validate config
    scenario = scenario_type(**yaml.safe_load(open(config_path)))

    # Create seeds
    if seed: 
        random.seed(seed)
    else:
        seed = random.randint(0, 2**32)
        random.seed(seed)
    random_seeds = [random.randint(0, 2**32) for _ in range(n_datasets)]

    # Run simulations
    simulations = []
    for ix in tqdm(range(n_datasets), desc="Simulating data"):
        ts, data = scenario(ix, random_seeds[ix], n_datasets) 
        simulations.append(Replicate(treeSequence=ts, data=data))

    # Output to file 
    print(f"Writing output ...")
    data = Simulations(scenario=scenario_type.__name__, config=scenario, seed=seed,
                       simulations=simulations) 
    with open(out_path, "w") as fh:
        fh.write(data.model_dump_json())

    print("Simulations complete.")


def parse_simulations(scenario: type, path: str):
    with open(path, "r") as fh:
        json_data = fh.read()
    return Simulations[scenario, scenario._data_class].model_validate_json(json_data)





