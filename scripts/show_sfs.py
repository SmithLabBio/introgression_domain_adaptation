#!/usr/bin/env python

import fire
import numpy as np

def pretty_print_matrix(path, normed=False):
    if path.endswith('.npz'):
        d = np.load(path)
        m = d["x"][0].squeeze(axis=-1)
    elif path.endswith('.npy'):
        m = np.load(path)
    max_digits = max(len(str(int(x))) for x in np.ravel(m))
    if not normed:
        for row in m:
            print(' '.join(f'{x:{max_digits}.0f}' for x in row))
    else:
        for row in m:
            print(' '.join(f"{x:{max_digits}.5f}"[1:] for x in row))

if __name__ == "__main__":
    fire.Fire(pretty_print_matrix)