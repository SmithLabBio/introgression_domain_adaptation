import numpy as np
import os
import fire

# path =   "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-test-sfs.npz"
# outdir = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-1-100-test-fsc" 

path =   "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-100-test-sfs.npz"
outdir = "/mnt/scratch/smithfs/cobb/popai/simulations/general-secondary-contact-ghost-1-100-test-fsc" 

os.makedirs(outdir, exist_ok=True)
data = np.load(path)

matrices = data["x"].squeeze()
labels = data["labels"]

pop0_labels = "\t".join(f"d0_{i}" for i in range(21))
pop1_labels = [f"d1_{i}" for i in range(21)]

np.set_printoptions(precision=2, suppress=True, linewidth=200)

def reformat(i):
    mat = matrices[i]
    for row in range(mat.shape[0] // 2):
        col = mat.shape[0] - 1 - row
        if not row == col:
            div_val = mat[row, col] / 2
            mat[row, col] = div_val
            mat[col, row] = div_val
    
    s = "1 observations\n" + "\t" + pop0_labels + "\n"
    for row in range(mat.shape[0]):
        s += pop1_labels[row] + "\t" + "\t".join(mat[row, :].astype(str))
        if not row == mat.shape[0] - 1:
            s += "\n"
    name = f"general-secondary-contact-1-100-test-{i}-label_{labels[i]}.txt"
    out = os.path.join(outdir, name)
    with open(out, "w") as f:
        f.write(s)

for i in range(matrices.shape[0]):
    reformat(i)
# reformat(0)
