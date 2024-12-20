#!/usr/bin/env python

import fire
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def pretty_print_matrix(m):
    max_digits = max(len(str(int(x))) for x in np.ravel(m))
    for row in m:
        print(' '.join(f"{x:{max_digits}.5f}"[1:] for x in row))

def slice(s):
    parts = s.split(':')
    
    def parse_part(part):
        if part == '':
            return None
        return int(part)
    
    if len(parts) == 1:
        return np.s_[parse_part(parts[0])]
    elif len(parts) == 2:
        return np.s_[parse_part(parts[0]):parse_part(parts[1])]
    elif len(parts) == 3:
        return np.s_[parse_part(parts[0]):parse_part(parts[1]):parse_part(parts[2])]
    else:
        raise ValueError("Invalid slice string")


def plot(path, out, ix=0, mean=False):
    if path.endswith('.npz'):
        d = np.load(path)
        if mean:
            m = np.mean(d["x"][slice(ix)].squeeze(axis=-1), axis=0)
        else:
            m = d["x"][slice(ix)].squeeze(axis=-1)
    elif path.endswith('.npy'):
        m = np.load(path)
    pretty_print_matrix(m)
    ax = sns.heatmap(m, norm=LogNorm(vmin=0.0000095, vmax=1), cbar=True, yticklabels=False, xticklabels=False, square=True)
    ax.tick_params(left=False, bottom=False)
    sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

    plt.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire(plot)