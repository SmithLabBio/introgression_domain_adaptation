from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

def predict(model, dataset):
    p = np.argmax(model.predict(dataset.snps), axis=1)
    r = confusion_matrix(np.argmax(dataset.migrationStates, axis=1), p)
    return r

def predict_afs(model, dataset):
    p = np.argmax(model.predict(dataset.afs), axis=1)
    r = confusion_matrix(np.argmax(dataset.migrationStates, axis=1), p)
    return r

def predict_afs_npy(model, dataset, labels, outfile):
    probs = model.predict(dataset)
    p = np.argmax(probs, axis=1)
    r = confusion_matrix(np.argmax(labels, axis=1), p)
    fpr, tpr, thresholds = roc_curve(np.argmax(labels, axis=1), probs[:,1])
    roc_auc = auc(fpr, tpr)
    data = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'aoc': [roc_auc for _ in range(len(fpr))]})
    data.to_csv(outfile)

    return r

def predict_npy(model, dataset, labels, outfile):
    probs = model.predict(dataset)
    p = np.argmax(probs, axis=1)
    r = confusion_matrix(np.argmax(labels, axis=1), p)
    fpr, tpr, thresholds = roc_curve(np.argmax(labels, axis=1), probs[:,1])
    roc_auc = auc(fpr, tpr)
    data = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'aoc': [roc_auc for _ in range(len(fpr))]})
    data.to_csv(outfile)
    return r
