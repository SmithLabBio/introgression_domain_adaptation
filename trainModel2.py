from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar

from src.simulation.data import Dataset 
from src.models.model1 import Model 
from src.lightning.lightningClassify import Lightning

    
trainDataset = Dataset("secondaryContact1/secondaryContact1-1000.json", 400, split=False)
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)

valDataset = Dataset("secondaryContact1/secondaryContact1-100-val.json", 400, split=False)
valLoader = DataLoader(valDataset, batch_size=64)

model = Lightning(Model(5, 50, 2, 50)) 
trainer = Trainer(min_epochs=5, max_epochs=10, log_every_n_steps=1, callbacks=[RichProgressBar()])
# trainer = Trainer(fast_dev_run=True, callbacks=[RichProgressBar()])

trainer.fit(model, trainLoader, valLoader)
trainer.save_checkpoint("secondaryContact1/spidna.ckpt")

