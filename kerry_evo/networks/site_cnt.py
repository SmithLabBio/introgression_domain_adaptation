#!/usr/bin/env python

import os
import fire
import tskit
from statistics import mean

from sim_wrapper.simulator import Simulations
from simulations.secondary_contact import SecondaryContact
from simulations.secondary_contact_ghost2 import GhostSecondaryContact


def cnt(Scenario, paths):
    if isinstance(paths, str):
        paths = [paths]
    Scenario = eval(Scenario)
    snp_cnt = []
    for path in paths:
        with open(path, "r") as fh:
            json_data = fh.read()
        sims = Simulations[Scenario, Scenario._data_class].model_validate_json(json_data)
        for i in sims:
            ts = i.treeSequence
            snp_cnt.append(ts.num_sites)
        print(f"Mean sites: {mean(snp_cnt)}")
    

if __name__ == "__main__":
    fire.Fire(cnt)

