from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("theta-predicted.csv")
plt.scatter(df["target"], df["predicted"])
plt.axline((0,0), slope=1)
plt.title("Theta")
plt.xlabel("Target")
plt.ylabel("Predicted")
plt.savefig("theta-predicted.png")
plt.clf()
