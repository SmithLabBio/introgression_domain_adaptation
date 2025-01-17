#!/usr/bin/env python

from pysam import VariantFile, set_verbosity
import numpy as np
import fire
from collections import Counter
from scipy.spatial.distance import euclidean
from sim_wrapper.genotype_matrix import sort_genotype_matrix

set_verbosity(0)

def get_length(n):
    cnt = 0
    for i in n:
        cnt += 1
    return cnt

def to_numpy(inpath):
    vcf_in = VariantFile(inpath) 
    n_samples = get_length(vcf_in.header.samples)
    n_sites = get_length(vcf_in)
    arr = np.zeros((2, n_samples, n_sites)) 
    for i, record in enumerate(vcf_in):
        for j, sample in enumerate(record.samples.values()):
            gt = sample["GT"]
            arr[0, j,i] = gt[0]
            arr[1, j,i] = gt[1]
    return arr

def to_major_minor(arr):
    """
    Reassign character states so the values represent the order of the frequencies in which they occur 
    """
    for col in range(arr.shape[2]):
        # Count the number of occurences of each value
        u = np.array(np.unique(arr[:,:,col], return_counts=True)).T
        # Get values sorted in order of frequency
        s = u[u[:, 1].argsort()[::-1]][:,0]
        # Create boolean mask for column where true indicates a value needs to be subsituted
        masks = []
        for i, val in enumerate(s):
            if i != val:
                masks.append((i, arr[:,:,col] == val))
        # Replace values using mask
        for m in masks:
            arr[:,:,col][m[1]] = m[0]

def run(inpath, outpath):
    arr = to_numpy(inpath)
    to_major_minor(arr)
    arr = np.array([
        sort_genotype_matrix(arr[0], euclidean),
        sort_genotype_matrix(arr[1], euclidean)])
    np.savez(outpath, arr)

if __name__ == "__main__":
    fire.Fire(run)