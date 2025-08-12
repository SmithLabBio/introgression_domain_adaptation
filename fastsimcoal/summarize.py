import glob
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

outdir = "/mnt/home/kc2824/scratch/fastsimcoal/general-secondary-contact-1-100-test-fsc-output"
# outdir = "/mnt/home/kc2824/scratch/fastsimcoal/general-secondary-contact-1-100-test-fsc-unlinked-output"

model_map = {"isolation": "0", "secondary_contact": "1"}
label_map = {"0": "isolation", "1": "secondary_contact"}

dfs = []
# For every output directory
for dir in os.listdir(outdir):
    # Split filename to get model, label, simulation replicate, and fsc2 replicate
    split = dir.split("-")
    fsc_rep = split[-1]
    model = split[-2]
    label = split[-3].split("_")[1]
    sim_rep = split[-4]
    lhood_path = os.path.join(outdir, dir, f"{model}.bestlhoods") # Path to the .bestlhoods file
    if os.path.exists(lhood_path):
        # Read likelihood output and add columns for fsc2 replicate, simulation replicate, model, and label
        df = pd.read_csv(lhood_path, sep="\t")
        df["fsc_rep"] = fsc_rep
        df["sim_rep"] = sim_rep
        df["model"] = model
        df["label"] = label
        dfs.append(df)
    else:
        print(dir, "does not have a .bestlhoods file")

# Concatenate all likelihood dataframes
df = pd.concat(dfs, ignore_index=True) 

true_model = [] 
predicted_model = []
for sim_rep, sim_rep_df in df.groupby("sim_rep"):
    estimates = []
    # For each model, get the highest likelihood value
    for model, model_df in sim_rep_df.groupby("model"):
        highest = model_df.loc[model_df['MaxEstLhood'].idxmin()]
        estimates.append(highest)
    estimates_df = pd.DataFrame(estimates)
    best = estimates_df.loc[estimates_df['MaxEstLhood'].idxmin()]
    true_model.append(int(best["label"]))
    predicted_model.append(int(model_map[best["model"]]))

cm =confusion_matrix(true_model, predicted_model)
print(cm)