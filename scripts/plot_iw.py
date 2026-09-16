"""
Plot IWCDAN class importance-weight trajectories to diagnose lambda.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

from extract_iw_weights import collect_iw_weights



def plot_iw_trajectories(iw_df, output_dir):
    g = sns.relplot(
        data=iw_df,
        x="epoch",
        y="iw_weight",
        hue="class",
        col="lambda_val",
        row="target_prop",
        units="replicate",
        estimator=None,
        kind="line",
        alpha=0.6,
        linewidth=1,
        height=4,
        aspect=1,
        facet_kws={"sharey": True},
    )
    g.set_titles("lambda_val = {col_name} | target_prop = {row_name}")
    g.set_axis_labels("Epoch", "Importance weight")
    g.tight_layout()
    g.savefig(os.path.join(output_dir,"iw_trajectories_by_lambda.png"), dpi=300, bbox_inches="tight")

def parse_arguments():

    parser = argparse.ArgumentParser(description="A script for organizing all results and making exploratory plots.")
    parser.add_argument("-i", "--iwcdan", help="Highest level input folder for results to analyze for IWCDAN results.")
    parser.add_argument("-o", "--output", help="Output folder for plots and summary statistics.")
    args = parser.parse_args()
    return args

def main():

    args = parse_arguments()

    iw_df = collect_iw_weights(args.iwcdan)
    iw_df.to_csv(args.output + "/iw_trajectory.csv", index=False)
    plot_iw_trajectories(iw_df, args.output)

if __name__ == "__main__":
    main()