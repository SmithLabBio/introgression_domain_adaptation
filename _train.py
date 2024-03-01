from torch.utils.data import DataLoader
from lightning import Trainer

from data import Dataset 
from _model import CNN

    
trainDataset = Dataset("secondaryContact1/secondaryContact1-1.json", 400, split=False)
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)

print(trainDataset[0])


# valDataset = Dataset("secondaryContact1/secondaryContact1-100-val.json", 400, split=True)
# valLoader = DataLoader(valDataset, batch_size=64)

# model = CNN(5, 50, 2) 
# # trainer = Trainer(min_epochs=5, max_epochs=5, log_every_n_steps=1)
# trainer = Trainer(fast_dev_run=True)
# trainer.fit(model, trainLoader, valLoader)
# # trainer.save_checkpoint("secondaryContact1/spidna.ckpt")

