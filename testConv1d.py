from torch.utils.data import DataLoader
from lightning import Trainer

from src.simulation.data import Dataset 
# from src.models.conv1d import Model 
from src.lightning.lightningClassify import Lightning


outdir = "out/conv1d-1/"
version = 0 
ckpt = "epoch=4-step=80"
ckptPath = f"{outdir}/lightning_logs/version_{version}/checkpoints/{ckpt}.ckpt"

testDataset = Dataset("secondaryContact1/secondaryContact1-100-test.json", 400, split=False)
testLoader = DataLoader(testDataset, batch_size=64)

nSamples = testDataset.simulations.config.nSamples 
model = Lightning.load_from_checkpoint(ckptPath)
trainer = Trainer(log_every_n_steps=1, default_root_dir=outdir)
trainer.test(model, testLoader)
print(model.confusionMatrix.compute())



# fig, ax = model.confmat.plot()
# fig.show()
# df = pd.DataFrame(cm, index=classes, columns=classes)
# # df.to_csv(f"{outDir}/confusionMatrix.csv")
# # with open(f"{outDir}/classification-report.txt", "w") as fh:
#     # fh.write(str(rep))
# sns.heatmap(df, annot=True, cmap="Blues")
# plt.savefig(f"{outDir}/confusion-matrix.png")



