from torch.utils.data import DataLoader
from lightning import Trainer

from thetaData import Dataset 
from thetaSpidnaModel import CNN


testDataset = Dataset("theta/theta1-100-test.json", 400, split=False)
testLoader = DataLoader(testDataset, batch_size=64)

model = CNN.load_from_checkpoint("theta/theta1.ckpt", n_blocks=5, n_features=50, n_outputs=1) 
trainer = Trainer(log_every_n_steps=1)
trainer.test(model, testLoader)


from matplotlib import pyplot as plt

plt.scatter(model.target, model.predicted)
plt.title("Prediction")
plt.xlabel("Target")
plt.ylabel("Predicted")
plt.savefig("theta/theta1.png")
