 tensorflow import keras
from tensorflow import Variable
# from keras.optimizers import Adam
from keras.optimizers.legacy import Adam
from keras.callbacks import ModelCheckpoint
from adapt.feature_based import CDAN
from keras.utils import to_categorical
from adapt.utils import UpdateLambda
from scipy.spatial.distance import euclidean
import os

from simulations.secondary_contact import SecondaryContact
from sim_wrapper.numpy_dataset import NumpySnpDataset

from util import plot_adapt_history, save_history
from models import getEncoder, getTask, getDiscriminator


n_snps = 100 
lambda_max = 10
epochs = 10 
learn_rate = 0.0001
disc_enc_learn_rate_ratio = 10

outdir = "/mnt/scratch/smithfs/cobb/popai/output/cdan-1/"
sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/" 

source =     NumpySnpDataset(SecondaryContact, f"{sim_dir}general-secondary-contact-1-1000-train.json",      "migration_state", n_snps, split=True, sorting=euclidean)
target =     NumpySnpDataset(SecondaryContact, f"{sim_dir}general-secondary-contact-ghost-1-100-train.json", "migration_state", n_snps, split=True,  sorting=euclidean)
target_val = NumpySnpDataset(SecondaryContact, f"{sim_dir}general-secondary-contact-ghost-1-100-val.json",   "migration_state", n_snps, split=True,  sorting=euclidean)

os.makedirs(f"{outdir}/checkpoints/", exist_ok=True)

checkpoint = ModelCheckpoint(f"{outdir}/checkpoints/{{epoch}}.hdf5", save_weights_only=True)

model = CDAN(
    lambda_=Variable(0.0), # Ignore Pycharm Warning
    encoder=getEncoder(shape=source.snps.shape[1:]), 
    task=getTask(), 
    discriminator=getDiscriminator(),
    optimizer = Adam(learn_rate),
    optimizer_enc = Adam(learn_rate),
    metrics=["accuracy"],
    callbacks=[
        checkpoint, 
        UpdateLambda(lambda_max=lambda_max)])

history = model.fit(
    X=source.snps, 
    y=to_categorical(source.labels, 2), 
    Xt=target.snps, 
    epochs=epochs, 
    batch_size=64, 
    validation_data=(target_val.snps, to_categorical(target_val.labels, 2)))

plot_adapt_history(history, outdir)
save_history(history.history.history, outdir)
