from tensorflow import keras
from tensorflow import Variable
from keras.optimizers.legacy import Adam
from adapt.feature_based import DANN
from adapt.utils import UpdateLambda
from scipy.spatial.distance import euclidean

from src.data.kerasSecondaryContactDataset import Dataset
from src.kerasPredictSigmoid import predict
from conv2d_models import getEncoder, getTask, getDiscriminator

snps = 1500 
source = Dataset("secondaryContact1/secondaryContact1-1000.json", snps, 
        sorting=euclidean, split=True)
target = Dataset("ghost1/ghost1-100.json", snps,
        sorting=euclidean, split=True)

model = DANN(
    lambda_=Variable(0.0),
    encoder=getEncoder(shape=source.shape), 
    task=getTask(), 
    discriminator=getDiscriminator(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
    optimizer=Adam(0.0001),
    callbacks=[UpdateLambda(lambda_max=0.1)]) 

history = model.fit(source.snps, source.migrationStates, target.snps, 
                    epochs=40, batch_size=64)
model.save("out/secondaryContact1-2d-dann.keras")

test = Dataset("ghost1/ghost1-test-100.json", snps,
        sorting=euclidean, split=True)

print(model.score(target.snps, target.migrationStates))
print(predict(model, test))