#!/usr/bin/env python

import fire
import numpy as np

def run(path, out):
    # Make an afs matrix consistent with the output of tskit
    # Fairly confident that this is correct for a symetric matrix, less confident about other scenarios 
    m = np.load(path)
    rows, cols = m.shape
    mid = rows // 2
    end = rows - 1
    for col in range(0, mid):
        row = end - col 
        m[col,row] = m[row,col] + m[col,row]
        m[row,col] = 0 
    np.save(out, m)
    print(out)

if __name__ == "__main__":
    fire.Fire(run)