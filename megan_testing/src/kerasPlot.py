from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os

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

def plotEncoded_afs(model, source, target, outputpath):
    Xs_enc_original = model.transform(source.afs)
    Xt_enc_original = model.transform(target.afs)
    X_original = np.concatenate((Xs_enc_original, Xt_enc_original))
    X_original_tsne = TSNE(2).fit_transform(X_original)
    plt.figure(figsize=(8, 6))
    plt.plot(X_original_tsne[:len(Xs_enc_original), 0], X_original_tsne[:len(Xs_enc_original), 1], '.', label="source")
    plt.plot(X_original_tsne[len(Xs_enc_original):, 0], X_original_tsne[len(Xs_enc_original):, 1], '.', label="target")
    plt.legend(fontsize=14)
    plt.title("Encoded Space tSNE")
    plt.savefig(outputpath)
    plt.close()

def plotEncoded_afs_npy(model, source, target, outputpath):
    Xs_enc_original = model.transform(source)
    Xt_enc_original = model.transform(target)
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

def plotEncoded_npy(model, source, target, outdir, outprefix):

    # encode data
    Xs_enc_original = model.transform(source)
    Xt_enc_original = model.transform(target)
    X_original = np.concatenate((Xs_enc_original, Xt_enc_original))

    ## tsne
    #X_original_tsne = TSNE(2).fit_transform(X_original)

    # pca
    pca_fit = PCA(2).fit(Xt_enc_original)
    X_original_pca = pca_fit.transform(X_original)
    #X_original_pca = PCA(2).fit_transform(X_original)

    ## plot tsne
    #plt.figure(figsize=(8, 6))
    #plt.plot(X_original_tsne[:len(Xs_enc_original), 0], X_original_tsne[:len(Xs_enc_original), 1], '.', label="source")
    #plt.plot(X_original_tsne[len(Xs_enc_original):, 0], X_original_tsne[len(Xs_enc_original):, 1], '.', label="target")
    #plt.legend(fontsize=14)
    #plt.title("Encoded Space tSNE")
    #plt.savefig(os.path.join(outdir, outprefix+"_tSNE.png"))
    #plt.close()
    #plt.figure()

    # plot pca
    plt.figure(figsize=(8, 6))
    plt.plot(X_original_pca[:len(Xs_enc_original), 0], X_original_pca[:len(Xs_enc_original), 1], '.', label="source")
    plt.plot(X_original_pca[len(Xs_enc_original):, 0], X_original_pca[len(Xs_enc_original):, 1], '.', label="target")
    plt.legend(fontsize=14)
    plt.title("Encoded Space PCA")
    plt.savefig(os.path.join(outdir, outprefix+"_PCA.png"))
    plt.close()
    plt.figure()

    # save text to file
    #np.savetxt(os.path.join(outdir, outprefix+"_tSNE.txt"), X_original_tsne)
    np.savetxt(os.path.join(outdir, outprefix+"_PCA.txt"), X_original_pca)
    np.savetxt(os.path.join(outdir, outprefix+"_encododed.txt"), X_original)

    ##  perform tests
    # split data
    source_pca = X_original_pca[:10000]
    target_pca = X_original_pca[-100:]
    #source_tsne = X_original_tsne[:10000]
    #target_tsne = X_original_tsne[-100:]

    # threshold
    threshold=0.01
    #threshold_pca = threshold / source_pca.shape[1] * 100
    #threshold_tsne = threshold / source_tsne.shape[1] * 100
    #threshold_all = threshold / Xs_enc_original.shape[1] * 100
    threshold_pca = threshold * 100
    #threshold_tsne = threshold * 100
    threshold_all = threshold * 100

    # find extremes
    more_extreme = np.logical_or(Xt_enc_original > np.percentile(Xs_enc_original, 100-threshold_all, axis=0), Xt_enc_original < np.percentile(Xs_enc_original, threshold_all, axis=0))
    more_extreme_pca = np.logical_or(target_pca > np.percentile(source_pca, 100-threshold_pca, axis=0), target_pca < np.percentile(source_pca, threshold_pca, axis=0))
    #more_extreme_tsne = np.logical_or(target_tsne > np.percentile(source_tsne, 100-threshold_tsne, axis=0), target_tsne < np.percentile(source_tsne, threshold_tsne, axis=0))

    # count extremes
    extreme_count = np.sum(more_extreme, axis=1)
    extreme_count_pca = np.sum(more_extreme_pca, axis=1)
    #extreme_count_tsne = np.sum(more_extreme_tsne, axis=1)

    # get violation indices
    violation_indices = np.where(extreme_count == Xs_enc_original.shape[1])[0]
    violation_indices_pca = np.where(extreme_count_pca == source_pca.shape[1])[0]
    #violation_indices_tsne = np.where(extreme_count_tsne == source_tsne.shape[1])[0]

    # write rsults to file
    with open(os.path.join(outdir, outprefix+"_test.txt"), 'w') as f:
        f.write(f"There were {len(violation_indices)} model violations.\n")
        np.savetxt(f, violation_indices, fmt='%d', delimiter='\t')
    
    # write rsults to file
    with open(os.path.join(outdir, outprefix+"_test_PCA.txt"), 'w') as f:
        f.write(f"There were {len(violation_indices_pca)} model violations.\n")
        np.savetxt(f, violation_indices_pca, fmt='%d', delimiter='\t')

    ## write rsults to file
    #with open(os.path.join(outdir, outprefix+"_test_tSNE.txt"), 'w') as f:
    #    f.write(f"There were {len(violation_indices_tsne)} model violations.\n")
    #    np.savetxt(f, violation_indices_tsne, fmt='%d', delimiter='\t')



