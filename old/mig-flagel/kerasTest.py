import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # silence tensorflow message 

import pickle
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv1D, AveragePooling1D, Dropout, Flatten, Dense  
from keras.models import load_model
from keras.utils import to_categorical
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def loadData(path, nSnps):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    migrationStates = data["migrationStates"]
    fullSnpMatrices = data["charMatrices"]
    snpMatrices = []
    for i in fullSnpMatrices:
        snpMatrices.append(i[:, :nSnps].transpose())
        # snpMatrices.append(i[:, :nSnps])
    snpMatrices = np.array(snpMatrices)
    return snpMatrices, migrationStates

model = load_model("keras-mig-weights.keras")
testSnpMatrices, testMigrationStates = loadData("mig-sims-100.pickle", 500)
predicted = np.argmax(model.predict(testSnpMatrices), axis=1)
confusion = confusion_matrix(testMigrationStates, predicted, labels=[0,1])
rep = classification_report(testMigrationStates, predicted, 
                            target_names=["none", "migration"], labels=[0,1])
print(confusion)
print(rep)
# df = pd.DataFrame.from_dict(dict(predicted=predicted, target=testPopSizes))
# df.to_csv("keras-mig-predicted.csv", index=False)

# # Plotting 
# from matplotlib import pyplot as plt
# import pandas as pd
# import numpy as np

# df = pd.read_csv("keras-theta-predicted.csv")
# plt.scatter(df["target"], df["predicted"])
# plt.axline((0,0), slope=1)
# plt.title("Theta")
# plt.xlabel("Target")
# plt.ylabel("Predicted")
# plt.savefig("keras-theta-predicted.png")
# plt.clf()
