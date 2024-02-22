#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # silence tensorflow message 

import fire
import pickle
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv1D, AveragePooling1D, Dropout, Flatten, Dense  
from keras.models import load_model
from keras.utils import to_categorical
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from data import getData

def run(testDataPath, weightsPath, outDir, channels, transpose):
    nSnps = 500
    data, snpMatrices, migrationStates = getData(testDataPath, nSnps, transpose=transpose)
    if channels == 1:
        nSamples = data["nSamples"] * 2 * 2 
    elif channels == 2:
        nSamples = data["nSamples"] * 2 
        snpMatrices = np.expand_dims(snpMatrices, axis=3)

    model = load_model(weightsPath)
    predicted = np.argmax(model.predict(snpMatrices), axis=1)

    classes = ["no migration", "migration"]
    cm = confusion_matrix(migrationStates, predicted, labels=[0,1])

    rep = classification_report(migrationStates, predicted, target_names=classes, 
            labels=[0,1])
    df = pd.DataFrame(cm, index=classes, columns=classes)
    df.to_csv(f"{outDir}/confusionMatrix.csv")
    with open(f"{outDir}/classification-report.txt", "w") as fh:
        fh.write(str(rep))
    print("\n*********************************************************************")
    print("Confusion Matrix\n")
    print(df)
    print("\n*********************************************************************")
    print("Classification Report\n")
    print(rep)

    sns.heatmap(df, annot=True, cmap="Blues")
    plt.savefig(f"{outDir}/confusion-matrix.png")

if __name__ == "__main__":
    fire.Fire(run)