from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import os

def plot_adapt_history(history, outdir):
    h = history.history.history
    fig, ax1 = plt.subplots()
    ax1.set_title("Training History")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.plot(h["accuracy"], color="darkgreen", label="Accuracy: %.2f"%h["accuracy"][-1])
    ax1.plot(h["disc_acc"], color="forestgreen", linestyle="dashed", label="Disc. Accuracy: %.2f"%h["disc_acc"][-1])
    ax1.plot(h["val_accuracy"], color="green", linestyle="dotted", label="Val. Accuracy: %.2f"%h["val_accuracy"][-1])
    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")
    ax2.plot(h["loss"], color="darkblue", label="Loss: %.2f"%h["loss"][-1])
    ax2.plot(h["disc_loss"], color="royalblue", linestyle="dashed", label="Disc. Loss: %.2f"%h["disc_loss"][-1])
    ax2.plot(h["val_loss"], color="blue", linestyle="dotted", label="Val Loss: %.2f"%h["val_loss"][-1])
    fig.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(f"{outdir}/loss-acc.png", bbox_inches="tight")
    plt.close()

def plot_tsne(model, source, target, outpath):
    Xs_enc_original = model.transform(source.x)
    Xt_enc_original = model.transform(target.x)
    X_original = np.concatenate((Xs_enc_original, Xt_enc_original))
    X_original_tsne = TSNE(2).fit_transform(X_original)
    plt.plot(X_original_tsne[:len(Xs_enc_original), 0], X_original_tsne[:len(Xs_enc_original), 1], '.', label="Source")
    plt.plot(X_original_tsne[len(Xs_enc_original):, 0], X_original_tsne[len(Xs_enc_original):, 1], '.', label="Target")
    plt.legend(fontsize=14)
    plt.title("Encoded Space tSNE")
    plt.savefig(outpath)
    plt.close()

def save_history(history, outdir):
    df = pd.DataFrame(history)
    df.to_csv(f"{outdir}/history.csv", index=False)

def save_confusion_matrix(outpath, labels, pred):
    cm =confusion_matrix(labels, np.argmax(pred, axis=1))
    df = pd.DataFrame(cm)
    df.to_csv(outpath, header=False, index=False)

def save_roc(outpath, labels, pred):
    fpr, tpr, thresholds = roc_curve(labels, pred[:,1])
    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    df.to_csv(outpath, index=False)


def plot_roc(data, outpath):
    """
    data: tuple(name, df_path]
    """
    for i in data:
        label = i[0]
        df = pd.read_csv(i[1])
        plt.plot(df['fpr'], df['tpr'], label=label)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(outpath)
    plt.close()

def get_auc(path):
    df = pd.read_csv(path)
    return auc(df["fpr"], df["tpr"])


# def mean_accuracy(dir):
#     dirs = os.listdir(dir)
#     for i in dirs:



# def predict(model, dataset, labels, outfile):
#     probs = model.predict(dataset)
#     p = np.argmax(probs, axis=1)
#     r = confusion_matrix(np.argmax(labels, axis=1), p)
#     fpr, tpr, thresholds = roc_curve(np.argmax(labels, axis=1), probs[:,1])
#     roc_auc = auc(fpr, tpr)
#     data = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'aoc': [roc_auc for _ in range(len(fpr))]})
#     data.to_csv(outfile)
#     return r

# def plotAdaptTrainingAcc(model, outputpath):
#     acc = model.history.history["accuracy"]
#     disc_acc = model.history.history["disc_acc"]
#     plt.plot(acc, label="Train acc - final value: %.3f"%acc[-1])
#     plt.plot(disc_acc, label="Disc acc - final value: %.3f"%disc_acc[-1])
#     plt.legend()
#     plt.xlabel("Epochs")
#     plt.ylabel("Acc")
#     plt.savefig(outputpath)
#     plt.close()

# def plotAdaptTrainingLoss(model, outputpath):
#     loss = model.history.history["loss"]
#     disc_loss = model.history.history["disc_loss"]
#     plt.plot(loss, label="Train Loss - final value: %.3f"%loss[-1])
#     plt.plot(disc_loss, label="Disc Loss - final value: %.3f"%disc_loss[-1])
#     plt.legend()
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.savefig(outputpath)
#     plt.close()

# def plotFineTrainingAcc(model, outputpath):
#     acc = model.history.history["accuracy"]
#     val_acc = model.history.history["val_accuracy"]
#     plt.plot(acc, label="Train Accuracy - final value: %.3f"%acc[-1])
#     plt.plot(val_acc, label="Validation Accuracy - final value: %.3f"%val_acc[-1])
#     plt.legend()
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.savefig(outputpath)
#     plt.close()

# def plotFineTrainingLoss(model, outputpath):
#     loss = model.history.history["loss"]
#     val_loss = model.history.history["val_loss"]
#     plt.plot(loss, label="Train Loss - final value: %.3f"%loss[-1])
#     plt.plot(val_loss, label="Validation Loss - final value: %.3f"%val_loss[-1])
#     plt.legend()
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.savefig(outputpath)
#     plt.close()

# def plotTrainingAcc(history, outputpath):
#     acc = history.history["accuracy"]
#     val_acc = history.history["val_accuracy"]
#     plt.plot(acc, label="Train acc - final value: %.3f"%acc[-1])
#     plt.plot(val_acc, label="Validation acc - final value: %.3f"%val_acc[-1])
#     plt.legend()
#     plt.xlabel("Epochs")
#     plt.ylabel("Acc")
#     plt.savefig(outputpath)
#     plt.close()

# def plotTrainingLoss(history, outputpath):
#     loss = history.history["loss"]
#     val_loss = history.history["val_loss"]
#     plt.plot(loss, label="Train Loss - final value: %.3f"%loss[-1])
#     plt.plot(val_loss, label="Validation Loss - final value: %.3f"%val_loss[-1])
#     plt.legend()
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.savefig(outputpath)
#     plt.close()