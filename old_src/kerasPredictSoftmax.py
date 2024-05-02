from sklearn.metrics import confusion_matrix
import numpy as np

def predict(model, dataset):
    pred = np.argmax(model.predict(dataset.snps), axis=1)
    r = confusion_matrix(np.argmax(dataset.migrationStates, axis=1), pred)
    return r
