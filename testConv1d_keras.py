from tensorflow import keras
from keras import models

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredict import predict

test = Dataset("secondaryContact1/secondaryContact1-test-500.json", 500, transpose=True)
model = models.load_model("secondaryContact1/keras_conv1d_model", compile=False)
print(predict(model, test))
