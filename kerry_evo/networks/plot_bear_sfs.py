import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors


pops = [
  "brown-alaska",
  "brown-us",
  "brown-hudson",
  "brown-asia",
  "brown-eurasia",
  "brown-eu",
  "brown-scandanavia"]

dir = "/mnt/scratch/smithlab/cobb/bears/filtered"

for i in pops:
    path = f"{dir}/brown-abc_{i}_cnts.npz"
    data = np.load(path)["x"].squeeze(axis=-1).sum(axis=0)    
    sns.heatmap(data=data, vmin=0, vmax=6000, cbar=False, yticklabels=False, xticklabels=False, 
            square=True, annot=True, fmt=".0f", annot_kws={"size":1})
    plt.savefig(f"{dir}/brown-abc_{i}_sfs.pdf")
    plt.clf()