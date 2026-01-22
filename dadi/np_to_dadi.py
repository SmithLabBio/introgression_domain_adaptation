import numpy as np
import dadi
import os
import fire


def convert(data_path, outdir):
    data = np.load(data_path)
    os.makedirs(outdir, exist_ok=True)

    mat = data["x"].squeeze()
    labels = data["labels"]

    for i in range(mat.shape[0]):
        sfs = dadi.Spectrum(mat[i, :, :], data_folded=True)
        name = f"general-secondary-contact-1-100-test-{i}-label_{labels[i]}.fs"
        sfs.to_file(os.path.join(outdir, name))

if __name__ == "__main__":
    fire.Fire(convert)