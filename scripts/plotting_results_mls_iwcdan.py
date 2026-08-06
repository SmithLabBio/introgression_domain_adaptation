import argparse
import pathlib
import os
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
import itertools


def parse_arguments():

    parser = argparse.ArgumentParser(description="A script for organizing all results and making exploratory plots.")
    parser.add_argument("-i", "--iwcdan", help="Highest level input folder for results to analyze for IWCDAN results.")
    parser.add_argument("-n", "--noiwcdan", help="Highest level input folder for results to analyze for noIWCDAN results.")
    parser.add_argument("-o", "--output", help="Output folder for plots and summary statistics.")
    args = parser.parse_args()
    return args

def collect_json(input):
    files = list(pathlib.Path(input).rglob("*.json"))
    dfs = []

    for f in files:
        with open(f) as temp:
            data = json.load(temp)
            df = pd.DataFrame([data])
        path_info = str(f).split("/")[-4]
        df["replicate"] = str(f).split("/")[-3]
        df["batch_size"] = path_info.split('.')[0].split('batch')[1]
        df["learning_rate"] = path_info.split('.learn_')[1].split('.')[0]
        df["enc_learning_rate"] = path_info.split('.enc_learn_')[1].split('.')[0]
        df["disc_learning_rate"] = path_info.split('.disc_learn_')[1].split('.')[0]
        df["lambda_val"] = path_info.split('.lambda_')[1].split('.target')[0]
        df["target_num"] = path_info.split('.target')[1].split('_')[1].split('.')[0]
        df["target_prop"] = float(path_info.split('.target')[1].split('_')[2])
        df["epochs"] = path_info.split('_')[-2]
        df["iwcdan"] = path_info.split('_')[-1]
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["replicate"] = pd.to_numeric(combined_df["replicate"])
    combined_df["batch_size"] = pd.to_numeric(combined_df["batch_size"])
    combined_df["learning_rate"] = pd.to_numeric(combined_df["learning_rate"])
    combined_df["enc_learning_rate"] = pd.to_numeric(combined_df["enc_learning_rate"])
    combined_df["disc_learning_rate"] = pd.to_numeric(combined_df["disc_learning_rate"])
    combined_df["lambda_val"] = pd.to_numeric(combined_df["lambda_val"])
    combined_df["target_num"] = pd.to_numeric(combined_df["target_num"])
    combined_df["epochs"] = pd.to_numeric(combined_df["epochs"])
    combined_df["iwcdan"] = combined_df["iwcdan"].astype(str)
    combined_df["iwcdan"].astype(str) == "True"
    return(combined_df)

def plot_metric(data: pd.DataFrame, y: str, fname: str):
    g = sns.catplot(
        data=data,
        x="target_prop",
        y=y,
        hue="iwcdan",
        col="lambda_val",
        kind="box",
        sharey=True,
        height=4,
        aspect=1,
    )
    g.set_titles("lambda_val = {col_name}")
    g.set_axis_labels("Target proportion", y.replace("_", " ").title())
    g.tight_layout()
    g.savefig(fname, dpi=150, bbox_inches="tight")
    return g

def plot_results(iwcdan, no_iwcdan, output_dir):

    print(iwcdan.head())
    print(no_iwcdan.head())

    df = pd.concat([iwcdan, no_iwcdan], ignore_index=True)

    param_cols = [
        "batch_size",
        "learning_rate",
        "enc_learning_rate",
        "disc_learning_rate",
        "lambda_val",
        "target_num",
        "target_prop",
        "epochs",
    ]

    nunique = df[param_cols].nunique().sort_values(ascending=False)
    varying = nunique[nunique > 1].index.tolist()
    if len(varying) > 2 and varying != ["target_prop", "lambda_val"]:
        exit("More than two parameters vary, and they are not just target_prop and lambda_val. Please check the input data.")

    p1 = plot_metric(
        df,
        "target_accuracy",
        "target_accuracy_plot.png",
    )
    
    p2 = plot_metric(
        df,
        "source_accuracy",
        "source_accuracy_plot.png",
    )
    
    out_name_target = f"accuracy_target_iwcdan.png"
    out_name_source = f"accuracy_source_iwcdan.png"
    p1.savefig(os.path.join(output_dir, out_name_target), dpi=300, bbox_inches="tight")
    p2.savefig(os.path.join(output_dir, out_name_source), dpi=300, bbox_inches="tight")


def main():

    args = parse_arguments()

    df_iwcdan = collect_json(args.iwcdan)
    df_noiwcdan = collect_json(args.noiwcdan)

    df_iwcdan.to_csv(os.path.join(args.output, "statistics_iwcdan.csv"))
    df_noiwcdan.to_csv(os.path.join(args.output, "statistics_noiwcdan.csv"))

    plot_results(df_iwcdan, df_noiwcdan, args.output)





if __name__ == "__main__":
    main()