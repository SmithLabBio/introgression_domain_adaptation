import glob
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import math
import sys
import numpy as np
from scipy.stats import chi2

# uncomment one of the following lines to set the output directory
outdir = "/mnt/scratch/smithlab/megan/da_revision/fastsimcoal_output/general-secondary-contact-1-100-test-fsc-unlinked-output"
# outdir = "/mnt/scratch/smithlab/megan/da_revision/fastsimcoal_output/general-secondary-contact-ghost-1-100-test-fsc-unlinked-output"

# Map confusion matrix row/col indices to model name
model_map = {"isolation": "0", "secondary_contact": "1"}
label_map = {"0": "isolation", "1": "secondary_contact"}

dfs = []
# Iterate over all directories in outdir
for dir in os.listdir(outdir):
    model = dir.split("-")[-2].strip('_v2')  # Extract model from directory name 
    fsc_replicate = dir.split("-")[-1]  # Extract replicate from directory name
    label = dir.split("-")[-3].split("_")[1]  # Extract label from directory name
    sim_replicate = dir.split("-")[-4]  # Extract simulation replicate from directory name
    dir_path = os.path.join(outdir, dir)
    lhood_path = os.path.join(dir_path, f"{dir.split('-')[-2]}.bestlhoods")  # Path to .bestlhoods file
    # Read the .bestlhoods file
    df = pd.read_csv(lhood_path, sep="\t")
    # Calculate the number of model parameters (columns - 2)
    k = df.shape[1] - 2
    # Extract the maximum likelihood estimate
    max_lhood = df["MaxEstLhood"].max()
    # Calculate AIC
    aic = 2 * k - 2 *(max_lhood/math.log10(math.exp(1)))
    # output
    df = pd.read_csv(lhood_path, sep="\t")
    df["fsc_rep"] = fsc_replicate
    df["sim_rep"] = sim_replicate
    df["model"] = model
    df["label"] = label
    df["aic"] = aic
    dfs.append(df)

# Concatenate all likelihood dataframes
df = pd.concat(dfs, ignore_index=True) 
df.sort_values(by=["sim_rep", "model", "label", "fsc_rep"], inplace=True)
df = df[["sim_rep", "model", "label", "fsc_rep", "MaxEstLhood", "MaxObsLhood", "aic", 'POPSIZE', 'TDIV', 'TMIG', 'RMIG']]
outputfile = f"{outdir.split('/')[-1]}_v2.csv"
df.to_csv(outputfile, na_rep="NA", index=False)

true_model = [] 
predicted_model = []
predicted_model_lrt = []
# Group by simulation replicate
for sim_rep, sim_rep_df in df.groupby("sim_rep"):
    best_estimates = []
    # For each simulation replicate group by model
    for model, model_df in sim_rep_df.groupby("model"):
        # Get row with the maximum log-likeihood
        best_estimate = model_df.loc[model_df['MaxEstLhood'].idxmax()]
        best_estimates.append(best_estimate)
        if model=="isolation":
            log10_null = best_estimate['MaxEstLhood']
        if model=="secondary_contact":
            log10_alt = best_estimate['MaxEstLhood']

    # perform LRT
    lnL_null = log10_null/math.log10(math.exp(1))
    lnL_alt  = log10_alt /math.log10(math.exp(1))
    df = 1 # degrees of freedom
    LR = 2 * (lnL_alt - lnL_null)
    p_value = chi2.sf(LR, df)
    # Make dataframe of best estimates for each model
    best_estimates_df = pd.DataFrame(best_estimates)
    # Get the model with the lowest AIC
    best_model = best_estimates_df.loc[best_estimates_df['aic'].idxmin()]
    # Determine the best model based on LRT
    if p_value < 0.01:
        best_model_lrt = best_estimates_df.loc[best_estimates_df['model']=='secondary_contact']
    else:
        best_model_lrt = best_estimates_df.loc[best_estimates_df['model']=='isolation']
    # Append true and predicted model to lists
    true_model.append(best_model["label"])
    predicted_model.append(model_map[best_model["model"]]) # Ignore pylance warning
    assert len(best_model_lrt) == 1, "Expected exactly one best model"
    model_name = best_model_lrt.at[best_model_lrt.index[0], "model"]
    predicted_model_lrt.append(model_map[model_name])

cm =confusion_matrix(true_model, predicted_model)
cm_lrt =confusion_matrix(true_model, predicted_model_lrt)
print("Confusion Matrix based on AIC:")
print(cm)
print("Confusion Matrix based on LRT:")
print(cm_lrt)