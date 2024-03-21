from torch.utils.data import DataLoader
from lightning import Trainer

from data.dataset import Dataset 
from src.models.spidna import Model 
from src.lightning.lightningClassify import Lightning



testDataset = Dataset("secondaryContact1/secondaryContact1-100-test.json", 400, split=False)
testLoader = DataLoader(testDataset, batch_size=64)

nSamples = testDataset.simulations.config.nSamples 
model = Model.load_from_checkpoint("secondaryContact1/spidna.ckpt", n_blocks=5, n_features=50, n_outputs=2) 
trainer = Trainer(log_every_n_steps=1)
trainer.test(model, testLoader)
print(model.confusionMatrix.compute())
# print(model.testAccuracy.compute())


# fig, ax = model.confmat.plot()
# fig.show()

# df = pd.DataFrame(cm, index=classes, columns=classes)
# # df.to_csv(f"{outDir}/confusionMatrix.csv")
# # with open(f"{outDir}/classification-report.txt", "w") as fh:
#     # fh.write(str(rep))
# sns.heatmap(df, annot=True, cmap="Blues")
# plt.savefig(f"{outDir}/confusion-matrix.png")



