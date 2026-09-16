"""
Extract the per-epoch IWCDAN class importance weights directly from saved
.hdf5 checkpoints, without rebuilding the model.

The weights live at: top_level_model_weights/iw_class_weights:0
"""
import json
import pathlib
import re

import h5py
import numpy as np
import pandas as pd

import sys

IW_DATASET = "top_level_model_weights/iw_class_weights:0"


def _epoch_from_filename(fname, epoch_regex=r"(\d+)"):
    m = re.search(epoch_regex, pathlib.Path(fname).name)
    if not m:
        raise ValueError(f"Couldn't parse an epoch number from {fname!r}")
    return int(m.group(1))


def collect_iw_weights(input, checkpoint_pattern="*.hdf5", epoch_regex=r"(\d+)"):

    files = list(pathlib.Path(input).rglob("*.json"))
    rows = []

    for f in files:
        path_info = str(f).split("/")[-4]
        replicate = str(f).split("/")[-3]

        params = dict(
            replicate=pd.to_numeric(replicate),
            batch_size=pd.to_numeric(path_info.split(".")[0].split("batch")[1]),
            learning_rate=pd.to_numeric(path_info.split(".learn_")[1].split(".")[0]),
            enc_learning_rate=pd.to_numeric(path_info.split(".enc_learn_")[1].split(".")[0]),
            disc_learning_rate=pd.to_numeric(path_info.split(".disc_learn_")[1].split(".")[0]),
            lambda_val=pd.to_numeric(path_info.split(".lambda_")[1].split(".target")[0]),
            target_num=pd.to_numeric(path_info.split(".target")[1].split("_")[1].split(".")[0]),
            target_prop=float(path_info.split(".target")[1].split("_")[2]),
            epochs=pd.to_numeric(path_info.split("_")[-2]),
            iwcdan=path_info.split("_")[-1] == "True",
        )

        checkpoints_dir = f.parent.parent / "checkpoints"
        ckpt_files = sorted(checkpoints_dir.glob(checkpoint_pattern))
        if not ckpt_files:
            print(f"[skip] no checkpoints found in {checkpoints_dir}")
            continue

        for ckpt in ckpt_files:
            try:
                epoch = _epoch_from_filename(ckpt, epoch_regex)
                with h5py.File(ckpt, "r") as h5:
                    if IW_DATASET not in h5:
                        print(f"[skip] {IW_DATASET!r} not found in {ckpt}")
                        break
                    iw = np.array(h5[IW_DATASET])
            except Exception as e:
                print(f"[skip] {ckpt}: {e}")
                continue

            for c, w in enumerate(iw):
                rows.append({**params, "epoch": epoch, "class": c, "iw_weight": w})

    if not rows:
        raise RuntimeError("No iw_class_weights found anywhere under " + str(input))

    df = pd.DataFrame(rows)
    return df.sort_values(
        ["lambda_val", "replicate", "epoch", "class"]
    ).reset_index(drop=True)
