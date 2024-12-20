from tensorflow import keras
from adapt.feature_based import CDAN
import numpy as np
import model1 as models

x = np.zeros((1, 20, 20, 1))
# model = CDAN(
#     encoder=models.getEncoder(shape=x.shape[1:]), 
#     task=models.getTask(), 
#     discriminator=models.getDiscriminator())
# model.fit(Xt=x, X=x, y=np.zeros((1,2)), epochs=0)


m = models.getEncoder(shape=x.shape[1:])
m.compile()
print(m.summary())