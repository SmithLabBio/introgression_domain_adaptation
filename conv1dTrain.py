from torch.utils.data import DataLoader
from lightning import Trainer

from data import Dataset 
from conv1dModel import CNN 

    
trainDataset = Dataset("secondaryContact1/secondaryContact1-1000.json", 400, split=False)
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)

valDataset = Dataset("secondaryContact1/secondaryContact1-100-val.json", 400, split=False)
valLoader = DataLoader(valDataset, batch_size=64)

model = CNN(trainDataset.simulations.config.nSamples * 4) 
trainer = Trainer(min_epochs=5, max_epochs=5, log_every_n_steps=1)
trainer.fit(model, trainLoader, valLoader)
trainer.save_checkpoint("conv1d.ckpt")

