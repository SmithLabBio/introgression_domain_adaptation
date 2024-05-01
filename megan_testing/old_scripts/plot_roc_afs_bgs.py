import pandas as pd
import matplotlib.pyplot as plt

results_original_test = pd.read_csv('results/afs_original_test_roc_bgs.tsv', sep=",")
results_original_bgs = pd.read_csv('results/afs_original_bgs_roc_bgs.tsv', sep=",")
results_cdan_test = pd.read_csv('results/afs_cdan_test_roc_bgs.tsv', sep=",")
results_cdan_bgs = pd.read_csv('results/afs_cdan_bgs_roc_bgs.tsv', sep=",")

# Assuming df1, df2, df3, df4 are your DataFrames
dfs = [results_original_test, results_original_bgs, results_cdan_test, results_cdan_bgs]
colors = ['b', 'g', 'r', 'c']  # You can adjust colors as needed
names = ["original (test)", "original (bgs)", 'CDAN (test)', 'CDAN (bgs)']
line_styles = ['-.', '--', '--', '--'] 

plt.figure(figsize=(8, 6))

for i, df in enumerate(dfs):
    plt.plot(df['fpr'], df['tpr'], color=colors[i], linestyle=line_styles[i], lw=2, label=names[i])

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.savefig("results/AFS_ROC_bgs.png")