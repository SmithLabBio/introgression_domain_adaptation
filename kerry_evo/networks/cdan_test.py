import fire
from tensorflow import keras
from adapt.feature_based import CDAN
from keras.utils import to_categorical
from kerry_evo.networks.snp_model1 import getEncoder, getTask, getDiscriminator
from scipy.spatial.distance import euclidean

from simulations.secondary_contact import SecondaryContact
from sim_wrapper.numpy_dataset import NumpySnpDataset
from util import save_confusion_matrix, save_roc


n_snps = 100 
epoch = 10

outdir = "/mnt/scratch/smithfs/cobb/popai/output/cdan-1/"
sim_dir = "/mnt/scratch/smithfs/cobb/popai/simulations/" 

target_test = NumpySnpDataset(SecondaryContact, f"{sim_dir}general-secondary-contact-ghost-1-100-test.json", "migration_state", n_snps, split=True,  sorting=euclidean)

model = CDAN(
    encoder=getEncoder(shape=target_test.snps.shape[1:]), 
    task=getTask(), 
    discriminator=getDiscriminator())

model.fit(Xt=target_test.snps, X=target_test.snps, y=to_categorical(target_test.labels), epochs=0)
model.load_weights(f"{outdir}/checkpoints/{10}.hdf5")
target_pred = model.predict(target_test.snps)
save_confusion_matrix(f"{outdir}/cm.csv", target_test.labels, target_pred)
save_roc(f"{outdir}/roc.csv", target_test.labels, target_pred)