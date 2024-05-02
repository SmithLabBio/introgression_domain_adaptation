from torch.utils.data import DataLoader
from lightning import Trainer

from src.simulation.data import Dataset 
from src.models.model1 import Model 
from src.lightning.lightningClassify import Lightning


testDataset = Dataset("secondaryContact1/secondaryContact1-100-test.json", 400, split=False)
testLoader = DataLoader(testDataset, batch_size=64)
m = model=Model(5, 50, 2, 50)
model = Lightning("secondaryContact1/spidna.ckpt", model=m)
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



