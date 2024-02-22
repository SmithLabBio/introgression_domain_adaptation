import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("mig-sims-1000.pickle", "rb") as fh:
    data = pickle.load(fh)


plt.scatter(data["migrationRates"], data["summaryStats"]["fst"])
plt.xlabel("Migration Rate")
plt.ylabel("Fst")
plt.savefig("fst-plot.png")
plt.clf()


plt.scatter(data["migrationRates"], data["summaryStats"]["dxy"])
plt.xlabel("Migration Rate")
plt.ylabel("dxy")
plt.savefig("dxy-plot.png")
plt.clf()