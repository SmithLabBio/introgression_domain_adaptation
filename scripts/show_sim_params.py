#!/usr/bin/env python

import fire

from sim_wrapper.simulator import Simulations
from simulations.secondary_contact import SecondaryContact
from simulations.secondary_contact_ghost2 import GhostSecondaryContact


def convert(scenario, path):
    scenario = eval(scenario)
    with open(path, "r") as fh:
        json_data = fh.read()
    sims = Simulations[scenario, scenario._data_class].model_validate_json(json_data)
    for i in sims:
        print(i.data)
        

if __name__ == "__main__":
    fire.Fire(convert)





