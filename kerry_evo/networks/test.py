#!/usr/bin/env python

from tensorflow import keras
import fire
from adapt.feature_based import CDAN
from keras.utils import to_categorical
from scipy.spatial.distance import euclidean
from importlib import import_module
import json
from sklearn.metrics import accuracy_score

from simulations.secondary_contact import SecondaryContact
from simulations.secondary_contact_ghost import GhostSecondaryContact
from sim_wrapper.numpy_dataset import NumpySnpDataset, NumpyAfsDataset

from util import save_confusion_matrix, save_roc, plot_roc, plot_tsne, get_auc


def test(json_path, source_path, target_path, epoch):
    with open(json_path) as fh:
        d = json.load(fh)

    match d["DataType"]:
        case "NumpySnpDataset": 
            source = NumpySnpDataset(eval(d["SrcType"]), source_path, "migration_state", n_snps=d["n_snps"], split=True, sorting=euclidean)
            target = NumpySnpDataset(eval(d["TgtType"]), target_path, "migration_state", n_snps=d["n_snps"], split=True, sorting=euclidean)
        case "NumpyAfsDataset":
            source = NumpyAfsDataset(eval(d["SrcType"]), source_path, "migration_state", expand_dims=True)
            target = NumpyAfsDataset(eval(d["TgtType"]), target_path, "migration_state", expand_dims=True)
        case _: 
            quit("Invalid DataType argument")

    models = import_module(d["ModelFile"]) 

    model = CDAN(
        encoder=models.getEncoder(shape=source.x.shape[1:]), 
        task=models.getTask(), 
        discriminator=models.getDiscriminator())

    model.fit(Xt=source.x, X=source.x, y=to_categorical(source.labels), epochs=0)
    model.load_weights(f"{d['outdir']}/checkpoints/{epoch}.hdf5")
    source_pred = model.predict(source.x)
    target_pred = model.predict(target.x)
    save_confusion_matrix(f"{d['outdir']}/source-cm.csv", source.labels, source_pred)
    save_confusion_matrix(f"{d['outdir']}/target-cm.csv", target.labels, target_pred)
    save_roc(f"{d['outdir']}/source-roc.csv", source.labels, source_pred)
    save_roc(f"{d['outdir']}/target-roc.csv", target.labels, target_pred)
    plot_roc([
        ("Source", f"{d['outdir']}/source-roc.csv"), 
        ("Target", f"{d['outdir']}/target-roc.csv")],
        f"{d['outdir']}/roc.png")
    plot_tsne(model, source, target, f"{d['outdir']}/tsne.png")

    stats = dict(
        source_accuracy=accuracy_score(source.labels, source_pred), 
        target_accuracy=accuracy_score(target.labels, target_pred),
        source_auc=get_auc(f"{d['outdir']}/source-roc.csv"),
        target_auc=get_auc(f"{d['outdir']}/target-roc.csv")
    )

    with open(f"{d['outdir']}/stats.json") as fh:
        json.dump(stats, fh)

if __name__ == "__main__":
    fire.Fire(test)