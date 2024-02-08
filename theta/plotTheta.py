from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv("theta-predicted.csv")
plt.scatter(df["target"], df["predicted"])
plt.title("Theta")
plt.xlabel("Target")
plt.ylabel("Predicted")
plt.savefig("theta-predicted.png")
plt.clf()
