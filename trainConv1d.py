from torch.utils.data import DataLoader
from lightning import Trainer

from src.simulation.data import Dataset 
from src.models.conv1d import Model 
from src.lightning.lightningRegress import Lightning

    
trainDataset = Dataset("secondaryContact1/secondaryContact1-1000.json", 400, split=False)
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)

valDataset = Dataset("secondaryContact1/secondaryContact1-100-val.json", 400, split=False)
valLoader = DataLoader(valDataset, batch_size=64)

model = Lightning(Model(trainDataset.simulations.config.nSamples * 4)) 
trainer = Trainer(min_epochs=5, max_epochs=5, log_every_n_steps=1)
trainer.fit(model, trainLoader, valLoader)
trainer.save_checkpoint("secondaryContact1/conv1d.ckpt")

