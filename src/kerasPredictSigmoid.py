from sklearn.metrics import confusion_matrix
import numpy as np

def predict(model, dataset):
    prob = model.predict(dataset.snps)
    pred = np.where(prob > 0.5, 1, 0)
    r = confusion_matrix(dataset.migrationStates, pred)
    return r
