import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from utils.data_structures import read_gen_data, read_data

df_raw = read_data("data/Daten_Klingenthal_raw.txt")
print(df_raw)
df_gen = read_gen_data()
print(df_gen)

def roll_mean(arr:np.ndarray):
    return scipy.ndimage.uniform_filter1d(arr, 12)


plt.plot(roll_mean(df_raw["Durchfluss_m3s"]), label="raw", alpha=0.7, c="red", lw=0.5)
for col in df_gen.columns:
    plt.plot(roll_mean(df_gen[col]), label=col, c="grey", alpha=0.5, lw=0.2)
    # calculate mean

print(np.mean(df_raw))
print(np.mean(df_gen))

plt.savefig("x.png")