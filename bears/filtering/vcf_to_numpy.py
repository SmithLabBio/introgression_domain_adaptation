#!/usr/bin/env python

import dadi 
import fire
import pandas as pd
import numpy as np

def convert(vcf, popfile, outpath):
    df = pd.read_csv(popfile, header=None, sep=" ") 
    # Group samples from popfile by population assignment (column 1)
    groups = df.groupby(1).size() # number of samples per group
    pop0_size = groups.get("pop0", 0)*2 # 2 haploid copies per sample
    pop1_size = groups.get("pop1", 0)*2 # 2 haploid copies per sample
    dd = dadi.Misc.make_data_dict_vcf(vcf, popfile)
    fs = dadi.Spectrum.from_data_dict(dd, ['pop0', 'pop1'], polarized=False, mask_corners=True,
            projections=[pop0_size, pop1_size]) # If polarized is False, output is folded
    # SFS matrix has shape (pop0_size, pop1_size) 
    np.save(outpath, fs.data)

if __name__ == "__main__":
    fire.Fire(convert)



