import matplotlib.pyplot as plt
import numpy as np

from read_data import read_data
from setup import pegelname, tu_mediumblue, tu_red, tu_grey


raw_path = f"Daten_{pegelname}.txt"
raw = read_data(raw_path)

# find min and max y value and corresponding x value
max_value = max(raw["Durchfluss"])
max_index = raw["Durchfluss"].idxmax()
max_month = raw["Monat"].iloc[max_index]
min_value = min(raw["Durchfluss"])
min_index = raw["Durchfluss"].idxmin()
min_month = raw["Monat"].iloc[min_index]

# calculate empiral distribution and plot as histogram
plt.figure(figsize=(10, 5))
plt.hist(raw["Durchfluss"], bins=np.arange(0, max_value+0.1, 0.1), 
         density=False, 
         color=tu_mediumblue, label="Empirische Verteilung",
         lw=0.8, edgecolor="black", alpha=0.8)
plt.xticks(np.arange(0, max_value+1, 1), minor=False)
plt.xticks(np.arange(0, max_value+0.1, 0.1), minor=True)
plt.xlim(left=0, right=round(max_value, 1)+0.1)
plt.xlabel("Durchfluss [m³/s]")
plt.ylabel("empirische Häufigkeit")
plt.grid(which="minor", axis="y", color="grey", alpha=0.15)
plt.grid(which="major", axis="y", color="grey", alpha=0.75)
plt.grid(which="minor", axis="x", color="grey", alpha=0.15)
plt.grid(which="major", axis="x", color="grey", alpha=0.75)
plt.xlim(left=0)
plt.twinx() # add second axis with density
plt.ylabel("Dichte")
plt.savefig(f"images/{pegelname}_hist.png", dpi=300, bbox_inches="tight")
