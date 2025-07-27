import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors

plt.rcParams['xtick.major.size'] = 0
plt.rcParams['ytick.major.size'] = 0
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10 
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 15 
plt.rcParams["figure.titlesize"] = 18
plt.rcParams["axes.labelpad"] = 20 

renaming = { 
  "brown-abc": "ABC Islands",
  "brown-alaska": "Alaska",
  "brown-asia": "Asia",
  "brown-eurasia": "Eastern Europe",
  "brown-eu": "Europe",
  "brown-hudson": "Hudson Bay",
  "brown-scandanavia": "Scandinavia",
  "brown-us": "North America"}

reorder = [
  "brown-alaska",
  "brown-us",
  "brown-hudson",
  "brown-asia",
  "brown-eurasia",
  "brown-eu",
  "brown-scandanavia"]

root = "/mnt/scratch/smithlab/cobb/bears/predictions"

def get_df(path):
    df = pd.read_csv(path, header=None)
    df[["pop0", "pop1", "chrom", "rep"]] = df[0].str.split(r"(?<!NW)_",  expand=True)
    df = df[df["pop0"] == "brown-abc"]
    out_df = df.copy(deep=True)
    out_df["pop1"] = out_df["pop1"].replace(renaming)
    out_df = out_df.rename(columns={"pop1":"Population", "chrom":"Scaffold ID", "rep":"Replicate", 
            1:"Probability No Introgression", 2:"Probability Introgression"})
    out_df = out_df[["Population", "Scaffold ID", "Replicate", "Probability No Introgression", "Probability Introgression"]]
    df = df.groupby(by=["pop0", "pop1", "chrom"], as_index=False)[2].mean()
    df_wide = df.pivot(index="pop1", columns="chrom", values=2)
    df_wide = df_wide.reindex(reorder)
    df_wide = df_wide.rename(index=renaming)
    return df_wide, out_df


df1, out_df1 = get_df(f"{root}/batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.csv")
df2, out_df2 = get_df(f"{root}/batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.lambda_0.5.csv")
out_df1.insert(0, "Lambda", 0) 
out_df2.insert(0, "Lambda", 0.5)
out_df = pd.concat([out_df1, out_df2], ignore_index=False)
out_path = f"{root}/sim2-batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.combined.csv"
out_df.sort_values(["Lambda", "Population", "Scaffold ID", "Replicate"], inplace=True)
out_df.to_csv(out_path, index=False)
print(f"Saved to: {out_path}")

for name, data in out_df.groupby(by="Lambda"):
    print(f"Lambda: {name}")
    print(f"  Total mean: {data['Probability Introgression'].mean()}")
    print(f"  Total std: {data['Probability Introgression'].std()}")
    for n, d in data.groupby(by="Population"):
        print(f"  {n}")
        print(f"    Mean: {d['Probability Introgression'].mean()}")
        print(f"    Std: {d['Probability Introgression'].std()}")


def plot_full(thresh, path):
    
    df1_thresh = df1.map(lambda x: 1 if x>=thresh else 0 ) 
    df2_thresh = df2.map(lambda x: 1 if x>=thresh else 0 ) 
    
    import matplotlib.gridspec as gridspec
    cmap = "cividis"
    fig, axs = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 8))
    
    sns.heatmap(df1, ax=axs[0,0], cmap=cmap, vmin=0, vmax=1, cbar=False)
    sns.heatmap(df2, ax=axs[0,1], cmap=cmap, vmin=0, vmax=1, cbar=False)#, cbar_ax=cax)
    sns.heatmap(df1_thresh, ax=axs[1,0], cmap=cmap, vmin=0, vmax=1, cbar=False)
    sns.heatmap(df2_thresh, ax=axs[1,1], cmap=cmap, vmin=0, vmax=1, cbar=False)
    
    for i, ax in enumerate(fig.get_axes()):
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    axs[0,0].set_title("Mis-specified", pad=20)
    axs[0,1].set_title("Domain Adaptation", pad=20)
    axs[0,0].set_ylabel("Softmax Probabilities")
    axs[1,0].set_ylabel(f"{thresh} Softmax Threshold")

    # for i, ax in enumerate(fig.get_axes()):
    #     ax.text(-0.05, 1.02, f"{chr(97 + i)})", transform=ax.transAxes, size=14, weight="bold")

    plt.tight_layout(rect=[0, 0, .9, 1])

    cbar_ax = fig.add_axes([0.90, 0.52, 0.02, 0.41])
    fig.colorbar(axs[1,1].collections[0], cax=cbar_ax)

    # # Old code which put colorbar in middle    
    # cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    # fig.colorbar(axs[1,1].collections[0], cax=cbar_ax)

    fig.savefig(path)
    print(f"Saved to: {path}")


def plot_single_thresh(thresh, path):
    
    df1_thresh = df1.map(lambda x: 1 if x>=thresh else 0 ) 
    df2_thresh = df2.map(lambda x: 1 if x>=thresh else 0 ) 
    
    import matplotlib.gridspec as gridspec
    cmap = "cividis"
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 4))
    
    sns.heatmap(df1_thresh, ax=axs[0], cmap=cmap, vmin=0, vmax=1, cbar=False)
    sns.heatmap(df2_thresh, ax=axs[1], cmap=cmap, vmin=0, vmax=1, cbar=False)
    
    for i, ax in enumerate(fig.get_axes()):
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    axs[0].set_title("Mis-specified", pad=20)
    axs[1].set_title("Domain Adaptation", pad=20)
    axs[0].set_ylabel(f"{thresh} Softmax Threshold")

    # for i, ax in enumerate(fig.get_axes()):
    #     ax.text(-0.05, 1.02, f"{chr(97 + i)})", transform=ax.transAxes, size=14, weight="bold")

    plt.tight_layout(rect=[0, 0, .9, 1])


    # # Old code which put colorbar in middle    
    # cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    # fig.colorbar(axs[1,1].collections[0], cax=cbar_ax)

    fig.savefig(path)
    print(f"Saved to: {path}")


def plot_multi_thresh(thresh, path):
    
    import matplotlib.gridspec as gridspec
    cmap = "cividis"
    fig, axs = plt.subplots(len(thresh), 2, sharey=True, sharex=True, figsize=(10, 8))

    for i in range(0, len(thresh)):
        left = i * 2
        right = left + 1

        df1_thresh = df1.map(lambda x: 1 if x>=thresh[i] else 0) 
        df2_thresh = df2.map(lambda x: 1 if x>=thresh[i] else 0) 
    
        sns.heatmap(df1_thresh, ax=axs.flat[left],  cmap=cmap, vmin=0, vmax=1, cbar=False)
        sns.heatmap(df2_thresh, ax=axs.flat[right], cmap=cmap, vmin=0, vmax=1, cbar=False)
        axs.flat[left].set_ylabel("")
        axs.flat[right].yaxis.set_label_position("right")
        axs.flat[right].set_ylabel(f"Threshold {thresh[i]}", rotation=270)
    
    for i, ax in enumerate(fig.get_axes()):
        ax.set_xticklabels([])
        ax.set_xlabel('')
    
    for i, ax in enumerate(fig.get_axes()):
        ax.text(-0.04, 1.05, f"{chr(97 + i)})", transform=ax.transAxes, size=14, weight="bold")
    
    axs.flat[0].set_title("Mis-specified")
    axs.flat[1].set_title("Domain Adaptation")

    plt.tight_layout(rect=[0, 0, .9, 1])

    fig.savefig(path)
    print(f"Saved to: {path}")

# plot_full(0.5, f"{root}/sim2-batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.heatmap-0.5.pdf")
# plot_full(0.95, f"{root}/sim2-batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.heatmap-0.95.pdf")

plot_single_thresh(0.5, f"{root}/sim2-batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.heatmap-0.5.pdf")
plot_single_thresh(0.95, f"{root}/sim2-batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.heatmap-0.95.pdf")

plot_multi_thresh([0.5, 0.6, 0.7, 0.8], f"{root}/sim2-batch16.learn_1e-3.enc_learn_1e-3.disc_learn_1e-3.heatmap-supp-0.5.pdf")
