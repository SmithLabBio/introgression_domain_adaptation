from torch.utils.data import DataLoader
from lightning import Trainer

from src.simulation.data import Dataset 
from src.models.conv1d import Model 
from src.lightning.lightningClassify import Lightning


outDir = "out/conv1d-1/"

trainDataset = Dataset("secondaryContact1/secondaryContact1-1000.json", 400, split=False)
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)

valDataset = Dataset("secondaryContact1/secondaryContact1-100-val.json", 400, split=False)
valLoader = DataLoader(valDataset, batch_size=64)

model = Lightning(Model, nSamples=trainDataset.simulations.config.nSamples * 4) 
trainer = Trainer(max_epochs=10, log_every_n_steps=1, default_root_dir=outDir)
trainer.fit(model, trainLoader, valLoader)

