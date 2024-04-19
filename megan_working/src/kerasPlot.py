from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def plotEncoded(model, source, target, outputpath):
    Xs_enc_original = model.transform(source.snps)
    Xt_enc_original = model.transform(target.snps)
    X_original = np.concatenate((Xs_enc_original, Xt_enc_original))
    X_original_tsne = TSNE(2).fit_transform(X_original)
    plt.figure(figsize=(8, 6))
    plt.plot(X_original_tsne[:len(Xs_enc_original), 0], X_original_tsne[:len(Xs_enc_original), 1], '.', label="source")
    plt.plot(X_original_tsne[len(Xs_enc_original):, 0], X_original_tsne[len(Xs_enc_original):, 1], '.', label="target")
    plt.legend(fontsize=14)
    plt.title("Encoded Space tSNE")
    plt.savefig(outputpath)
    plt.close()

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