from tensorflow import keras
from keras import models
import numpy as np

from src.data.kerasSecondaryContactDataset import Dataset
from kerasPredictSigmoid import predict

test = Dataset("secondaryContact1/secondaryContact1-test-100.json", 500, transpose=True)
model = models.load_model("secondaryContact1/keras_conv1d_model")
# print(predict(model, test))

p = model.predict(test.snps)
print(np.where(p > 0.5, 1, 0))
    # p = np.argmax(model.predict(dataset.snps), axis=1)
    # r = confusion_matrix(np.argmax(dataset.migrationStates, axis=1), p)