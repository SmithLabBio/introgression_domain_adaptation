import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

renaming = { 
  "brown-abc": "ABC Islands",
  "brown-alaska": "Alaska",
  "brown-asia": "Asia",
  "brown-eurasia": "Eastern Europe",
  "brown-eu": "Europe",
  "brown-hudson": "Hudson Bay",
  "brown-scandanavia": "Scandinavia",
  "brown-us": "North America"
}

colors = {
    "brown-abc": "#1b9e77",
    "brown-alaska": "#d95f02",
    "brown-asia": "#7570b3",
    "brown-eurasia": "#666666",
    "brown-eu": "#66a61e",
    "brown-hudson": "#e6ab02",
    "brown-scandanavia": "#a6761d",
    "brown-us": "#e7298a"}

directory = "/mnt/home/kc2824/bears/filtering"
brown_files = [f for f in os.listdir(directory) if f.startswith("brown-")]
samples = []
for pop in renaming.keys():
    file_path = f"{directory}/{pop}.txt"
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()
        samples.append(dict(population=pop, biosample=line.split('/')[-1].split('.')[0]))

sample_df = pd.DataFrame(samples)
data_df = pd.read_csv("/mnt/home/kc2824/bears/data/bear-samples.csv")
df = sample_df.merge(data_df, left_on="biosample", right_on="BioSample ID", how="left")

# Adjust position of some samples to eliminate overlap
df.loc[df["biosample"] == "SAMN09907428", "Lon"] += 6 
df.loc[df["biosample"] == "SAMN32301302", "Lon"] += 6 
df.loc[df["biosample"] == "SAMN32301303", "Lon"] += 6 

# Map projection
transform = ccrs.NorthPolarStereo(-30)
# transform = ccrs.Orthographic(0, 90)
# transform = ccrs.PlateCarree()

# Map elements
ax = plt.axes(projection=transform)
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=(0.8, 0.8, 0.8))
ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=(1, 1, 1))
ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.01)

gl = ax.gridlines(
    linewidth=0.5,
    linestyle="--",
    alpha=1)
gl.xlocator = mticker.LinearLocator(numticks=12)

# lat0 = df["Lat"].min() - 50
# lat1 = df["Lat"].max() + 10
# lon0 = df["Lon"].min() - 5 
# lon1 = df["Lon"].max() + 5 
# ax.set_extent((lon0, lon1, lat0, lat1), crs=ccrs.PlateCarree())

# Plot extent control points
# lowLat = 0 
# lowLat = 40 
# ax.scatter([-180, -90, 0, 90], [lowLat]*4, s=0, transform=ccrs.PlateCarree())

# Plot data
for name, group in df.groupby("population"):
    population_name = renaming[str(name)]
    ax.plot(group["Lon"], group["Lat"], 'o', c=colors[str(name)], markersize=5, transform=ccrs.PlateCarree(), label=population_name)

# ax.legend(loc="upper right", handletextpad=-.5, markerscale=0.75, fontsize=7)
ax.legend(loc="lower right", ncols=4, handletextpad=-.5, markerscale=0.75, fontsize=7, framealpha=1)

plt.savefig("map.pdf", bbox_inches="tight")

ncbi_df = pd.read_csv("data/bear-samples-run-data.csv")
ncbi_df = ncbi_df.merge(df, how="inner", left_on="sample_accession", right_on="biosample")
ncbi_df["population"] = ncbi_df["population"].map(renaming)
ncbi_df = ncbi_df.rename(columns={"run_accession": "SRA Run ID", "Sample ID": "Publication Sample ID", "population": "Population"})
ncbi_df.to_csv("data/selected-for-analysis.csv", index=False, columns=["BioSample ID", "SRA Run ID", "Population", "Publication", "Publication Sample ID", "Lat", "Lon"])


