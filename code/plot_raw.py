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


plt.figure(figsize=(10, 5))

# plot data
plt.plot(raw["Monat"], raw["Durchfluss"], c=tu_mediumblue, linewidth=0.8, label="Rohdaten")

# plot extreme values as red scatter points
plt.axhline(y=max_value, c=tu_red, linestyle="--", linewidth=0.8, 
            label=f"Max: {max_month}: {max_value} m³/s")
plt.axhline(y=min_value, c=tu_grey, linestyle="--", linewidth=0.8, 
            label=f"Min: {min_month}: {min_value} m³/s")
plt.scatter(max_month, max_value, marker="o", facecolors='none', edgecolors=tu_red, s=30)
plt.scatter(min_month, min_value, marker="o", facecolors='none', edgecolors=tu_grey, s=30)

# plot config
plt.xlabel("Monat")
plt.ylabel("Durchfluss [m³/s]")
plt.xticks(raw["Monat"][::12], rotation=90)
plt.yticks(np.arange(0, max_value, 1), minor=False)
plt.yticks(np.arange(0, max_value, 0.25), minor=True)
plt.grid(which="major", axis="x", color="grey", alpha=0.15)
plt.grid(which="major", axis="y", color="grey", alpha=0.75)
plt.grid(which="minor", axis="y", color="grey", alpha=0.15)
plt.ylim(bottom=0)
plt.xlim(left=raw["Monat"].min(), right=raw["Monat"].max())
plt.legend(loc="upper right")
plt.savefig(f"images/{pegelname}_raw.png", dpi=300, bbox_inches="tight")