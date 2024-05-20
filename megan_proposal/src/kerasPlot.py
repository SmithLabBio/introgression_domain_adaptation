from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os

def plotTrainingAcc(model, outputpath):
    acc = model.history.history["accuracy"]
    disc_acc = model.history.history["disc_acc"]
    plt.plot(acc, label="Train acc - final value: %.3f"%acc[-1])
    plt.plot(disc_acc, label="Disc acc - final value: %.3f"%disc_acc[-1])
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.savefig(outputpath)
    plt.close()

def plotTrainingLoss(model, outputpath):
    loss = model.history.history["loss"]
    disc_loss = model.history.history["disc_loss"]
    plt.plot(loss, label="Train Loss - final value: %.3f"%loss[-1])
    plt.plot(disc_loss, label="Disc Loss - final value: %.3f"%disc_loss[-1])
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(outputpath)
    plt.close()

def getEncoded(model, source, target, outdir, outprefix):

    # encode data
    Xs_enc_original = model.transform(source)
    Xt_enc_original = model.transform(target)
    X_original = np.concatenate((Xs_enc_original, Xt_enc_original))

    # save text to file
    np.savetxt(os.path.join(outdir, outprefix+"_encododed.txt"), X_original)
