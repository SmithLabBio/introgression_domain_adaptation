import glob 
import statistics 

path = "/mnt/scratch/smithlab/cobb/bears/filtered"
files = [
  "brown-alaska",
  "brown-us",
  "brown-hudson",
  "brown-asia",
  "brown-eurasia",
  "brown-eu",
  "brown-scandanavia"]

counts = []
for i in files:
    cnt_within = [] 
    pattern = f"{path}/brown-abc_{i}_*site_count.txt" # Underscore after {i} is important!
    gl = glob.glob(pattern)
    for ix, j in enumerate(gl):
        cnt = int(open(j, 'r').readline().strip().split()[-1])
        cnt_within.append(cnt)
    counts.extend(cnt_within)
    # mean_within = statistics.mean(cnt_within)
    # std_within = statistics.stdev(cnt_within)
    # print(f"Mean within: {mean_within}")
    # print(f"Stdev within: {std_within}")


mean = statistics.mean(counts)
std = statistics.stdev(counts)
print(f"Mean: {mean}")
print(f"Stdev: {std}")