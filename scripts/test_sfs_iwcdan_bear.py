#!/usr/bin/env python

from tensorflow import keras
import fire
from adapt.feature_based import CDAN
from adapt.parameter_based import FineTuning
from keras.utils import to_categorical
from scipy.spatial.distance import euclidean
from importlib import import_module
import json
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import model1 as models
import glob
import re

from simulations.secondary_contact import SecondaryContact
from simulations.secondary_contact_ghost import GhostSecondaryContact
from sim_wrapper.numpy_dataset import NumpySnpDataset, NumpyAfsDataset

from util import save_confusion_matrix, save_roc, plot_roc, plot_tsne, get_auc, save_latent_space, save_predictions

from iw_cdan_v2 import IWCDAN, IWWeightUpdater



def test(dir, meth, source_path):

    source = np.load(source_path)
    target_array = np.load(f"{dir}/bear-secondary-contact-ghost-2-100-train-sfs-norm.subsampled.npz", allow_pickle=True)
    target = target_array["data"].item()

    checkpoint_files = glob.glob(f"{dir}/checkpoints/*.hdf5")
    print(dir)
    print(checkpoint_files)
    latest_checkpoint = max(
        checkpoint_files,
        key=lambda f: int(re.search(r'(\d+)\.hdf5$', f).group(1))
    )

    outdir = f"{dir}/summary"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if meth == "cdan":
        model = CDAN(
            encoder=models.getEncoder(shape=source["x"].shape[1:]), 
            task=models.getTask(), 
            discriminator=models.getDiscriminator())
        model.fit(Xt=source["x"], X=source["x"], y=to_categorical(source["labels"]), epochs=0)
    elif meth == "iwcdan":
        model = IWCDAN(
            encoder=models.getEncoder(shape=source["x"].shape[1:]),
            task=models.getTask(),
            discriminator=models.getDiscriminator(),
            n_classes=2)
        model.fit(Xt=source["x"], X=source["x"], y=to_categorical(source["labels"]), epochs=0)
        model.set_balance_weights(source["labels"])
    elif meth == "finetune":
        model = FineTuning(
            encoder=models.getEncoder(shape=source["x"].shape[1:]),
            task=models.getTask())
        model.fit(source["x"], to_categorical(source["labels"]), epochs=0)

    model.load_weights(latest_checkpoint)
    source_pred = model.predict(source["x"])
    target_pred = model.predict(target["x"])
    save_predictions(source["labels"], source_pred, f"{outdir}/source-predictions.csv")
    save_predictions(target["labels"], target_pred, f"{outdir}/target-predictions.csv")
    save_latent_space(model, source, target, f"{outdir}/latent-space.npz")
    save_confusion_matrix(f"{outdir}/source-cm.csv", source["labels"], source_pred)
    save_confusion_matrix(f"{outdir}/target-cm.csv", target["labels"], target_pred)
    save_roc(f"{outdir}/source-roc.csv", source["labels"], source_pred)
    save_roc(f"{outdir}/target-roc.csv", target["labels"], target_pred)
    plot_roc([
        ("Source", f"{outdir}/source-roc.csv"), 
        ("Target", f"{outdir}/target-roc.csv")],
        f"{outdir}/roc.png")

    plot_tsne(model, source, target, f"{outdir}/tsne.png")

    stats = dict(
        source_accuracy=accuracy_score(source["labels"], np.argmax(source_pred, axis=1)), 
        target_accuracy=accuracy_score(target["labels"], np.argmax(target_pred, axis=1)),
        source_auc=get_auc(f"{outdir}/source-roc.csv"),
        target_auc=get_auc(f"{outdir}/target-roc.csv")
    )

    with open(f"{outdir}/stats.json", "w") as fh:
        json.dump(stats, fh)

if __name__ == "__main__":
    fire.Fire(test)