from torch.utils.data import DataLoader
from lightning import Trainer

from src.data.thetaData import Dataset 
from src.thetaSpidna import Model 
from src.lightning.lightningClassify import Lightning

    
trainDataset = Dataset("theta/theta1-1000.json", 400, split=False)
trainLoader = DataLoader(trainDataset, batch_size=1, shuffle=True)

valDataset = Dataset("theta/theta1-100-val.json", 400, split=False)
valLoader = DataLoader(valDataset, batch_size=64)

model = Lightning(Model(5, 50, 1)) 
trainer = Trainer(min_epochs=5, max_epochs=5, log_every_n_steps=1)
# trainer = Trainer(fast_dev_run=True)
trainer.fit(model, trainLoader, valLoader)
trainer.save_checkpoint("theta/theta1.ckpt")

