import numpy as np
import os
import fire

path =   "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-test-sfs.npz"
outdir = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-test-fsc" 

data = np.load(path)

matrices = data["x"].squeeze()
labels = data["labels"]

pop0_labels = "\t".join(f"d0_{i}" for i in range(21))
pop1_labels = [f"d1_{i}" for i in range(21)]

for i in range(matrices.shape[0]):
    mat = matrices[i, :]
    s = "1 observations\n" + pop0_labels + "\n"
    for row in range(mat.shape[0]):
        col = mat.shape[0] - 1 - row
        div_val = mat[row, col] / 2
        mat[row, col] = div_val
        mat[col, row] = div_val
        s += pop1_labels[row] + "\t" + "\t".join(mat[row, :].astype(str)) + "\n"
    name = f"general-secondary-contact-1-100-test-{i}-label_{labels[i]}.txt"
    out = os.path.join(outdir, name)
    with open(out, "w") as f:
        f.write(s)
