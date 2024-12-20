import msprime as mp
import tskit as ts
import seaborn as sns
import matplotlib.pyplot as plt

#TODO: A better/safer way to keep track of samples within populations

dem = mp.Demography()
a = dem.add_population(name="a", initial_size=1000)
b = dem.add_population(name="b", initial_size=1000)
c = dem.add_population(name="c", initial_size=1000)
dem.add_population_split(time=10000, derived=["b", "c"], ancestral="a")

ts = mp.sim_ancestry(samples=dict(b=2, c=4), demography=dem, sequence_length=1000) 
mts = mp.sim_mutations(ts, rate=1e-2)

a_samples = mts.samples(b.id)
b_samples = mts.samples(c.id)

print(len(a_samples))
print(len(b_samples))

afs = mts.allele_frequency_spectrum(sample_sets=[a_samples, b_samples])
print(afs.shape)

sns.heatmap(afs)
plt.savefig("test.png")

