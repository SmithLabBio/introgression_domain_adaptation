#!/usr/bin/env python
import fire
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pc(input, output):
    df = pd.read_csv(input, sep='\t')
    melted = df.melt(id_vars=["PC1", "Group"], value_vars=["PC2", "PC3"], var_name="PC", value_name="Value")
    g = sns.FacetGrid(melted, col="PC", hue="Group")
    g.map(sns.scatterplot, "PC1", "Value")
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(f"{output}.pdf", bbox_inches='tight')



if __name__ == '__main__':
    fire.Fire(plot_pc)