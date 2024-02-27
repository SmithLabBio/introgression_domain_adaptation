from torch.utils.data import DataLoader
from lightning import Trainer

from data import Dataset 
from conv1dModel import CNN 

    
trainDataset = Dataset("../secondaryContact1/secondaryContact1-1000.json", 400, split=False)
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)

model = CNN(trainDataset.simulations.config.nSamples * 4) 
trainer = Trainer(min_epochs=5, max_epochs=5)
trainer.fit(model, trainLoader)
trainer.save_checkpoint("conv1d.ckpt")

