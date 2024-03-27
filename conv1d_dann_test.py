from tensorflow import keras
from keras import models

from src.data.kerasSecondaryContactDataset import Dataset
from kerasPredictSigmoid import predict

test = Dataset("ghost1/ghost1-test-100.json", 100, transpose=True)
model = models.load_model("ghost1/conv1d_dann_model", compile=False)
print(predict(model, test))