from tensorflow import keras
from keras import models

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredict import predict

test = Dataset("ghost1/ghost1-test-100.json", 100, transpose=True)
model = models.load_model("ghost1/conv1d_cdan_model", compile=False)
print(predict(model, test))