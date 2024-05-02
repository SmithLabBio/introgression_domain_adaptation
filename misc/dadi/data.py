import sys, os
sys.path.append(os.path.abspath(".."))

from src.data.simulation import Simulations
import dadi
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

with open("../secondaryContact3/secondaryContact3-2.json", "rb") as fh:
    data = fh.read()
sims = Simulations.model_validate_json(data)
ts = sims[0].treeSequence
nSamples = len(ts.samples())
pop1 = list(range(0, nSamples//2))
pop2 = list(range(nSamples//2, nSamples))
sfs = ts.allele_frequency_spectrum([pop1, pop2], polarised=True, span_normalise=False)
fs = dadi.Spectrum(sfs)





# plt.figure()
# plt.imshow(fs, interpolation='nearest', norm=Normalize(vmin=sfs.min()*(1-1e-3), vmax=sfs.max()*(1+1e-3)))
# plt.colorbar()
# plt.savefig(f"sft-{i}.png



