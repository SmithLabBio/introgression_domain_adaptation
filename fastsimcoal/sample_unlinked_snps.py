import numpy as np
import os
import fire
from simulations.secondary_contact import SecondaryContact
from simulations.simulator import Simulations

import tskit
import numpy as np


# path =   "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-test.json"
# outdir = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-test-fsc-unlinked" 

path =   "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-100-test.json"
outdir = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-100-test-fsc-unlinked" 

os.makedirs(outdir, exist_ok=True)

min_distance = 500

with open(path, "r") as fh:
    json_data = fh.read()
simulations = Simulations[SecondaryContact, SecondaryContact._data_class].model_validate_json(json_data)

for rep in range(len(simulations)):
    label = simulations[rep].data.migration_state
    ts = simulations[rep].treeSequence
    
    print(ts.num_sites)
    
    # Drop sites within n bases of each other
    last_selected = ts.site(0).position
    sites_to_drop = []
    for i in range(1, ts.num_sites):
        site = ts.site(i)
        if site.position - last_selected >= min_distance:
            last_selected = site.position
        else:
            sites_to_drop.append(i)
    ts = ts.delete_sites(sites_to_drop)
    
    # Compute allele frequency spectrum
    n_samples = len(ts.samples())
    pop1 = list(range(0, n_samples//2))
    pop2 = list(range(n_samples//2, n_samples))
    afs = ts.allele_frequency_spectrum(sample_sets=[pop1, pop2], span_normalise=False, polarised=False)

    # Fix AFS cells on the diagonal 
    for row in range(afs.shape[0] // 2):
        col = afs.shape[0] - 1 - row
        if not row == col:
            div_val = afs[row, col] / 2
            afs[row, col] = div_val
            afs[col, row] = div_val

    # Output afs to file 
    pop0_labels = "\t".join(f"d0_{i}" for i in range(20 + 1))
    pop1_labels = [f"d1_{i}" for i in range(20 + 1)]
    s = "1 observations\n" + "\t" + pop0_labels + "\n"
    for row in range(afs.shape[0]):
        s += pop1_labels[row] + "\t" + "\t".join(afs[row, :].astype(str))
        if not row == afs.shape[0] - 1:
            s += "\n"
    name = f"general-secondary-contact-1-100-test-{rep}-label_{label}.txt"
    out = os.path.join(outdir, name)
    with open(out, "w") as f:
        f.write(s)
