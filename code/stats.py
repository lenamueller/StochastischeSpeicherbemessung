import numpy as np
import pandas as pd
import scipy

from setup import pegelname
from read_data import read_data


raw_path = f"Daten_{pegelname}.txt"
raw = read_data(raw_path)

# min
min_value = min(raw["Durchfluss"])
min_index = raw["Durchfluss"].idxmin()
min_month = raw["Monat"].iloc[min_index]
print(f"Min: {min_month}: {min_value} m³/s")

# max 
max_value = max(raw["Durchfluss"])
max_index = raw["Durchfluss"].idxmax()
max_month = raw["Monat"].iloc[max_index]
print(f"Max: {max_month}: {max_value} m³/s")

print("---------------------------------------------")

# first central moments
mom_1 = raw["Durchfluss"].mean()
print(f"First central moment: {np.round(mom_1, 3)} m³/s")

# second central moment
mom_2 = raw["Durchfluss"].var()
print(f"Second central moment: {np.round(mom_2, 3)} m³/s")

# third central moment
mom_3 = scipy.stats.skew(raw["Durchfluss"], bias=True)
print(f"Third central moment: {np.round(mom_3, 3)}")

# fourth central moment
mom_4 = scipy.stats.kurtosis(raw["Durchfluss"], bias=True)
print(f"Fourth central moment: {np.round(mom_4, 3)}")

print("---------------------------------------------")

# calculate biased standard deviation
std_value = np.sqrt(mom_2)
print(f"Standard deviation: {np.round(std_value, 3)} m³/s")

# calculate unbiased standard deviation
std_value = np.sqrt(mom_2 * (raw.shape[0] / (raw.shape[0] - 1)))
print(f"Standard deviation unbiased: {np.round(std_value, 3)} m³/s")

def skewness(x):
    mean = x.mean()
    std = x.std()
    n = len(x)
    skew = np.sum(((x - mean)/std)**3) / n
    skew_unbiased = skew * n/(n-1) * (n-1)/(n-2)
    return skew, skew_unbiased

def kurtosis(x):
    mean = x.mean()
    std = x.std()
    n = len(x)
    kurt = np.sum(((x - mean)/std)**4) / n - 3
    kurt_unbiased = kurt * n/(n-1) * (n-1)/(n-2) * (n-2)/(n-3)
    return kurt, kurt_unbiased

print(f"Skewness: {np.round(skewness(raw['Durchfluss'])[0], 3)}")
print(f"Skewness unbiased: {np.round(skewness(raw['Durchfluss'])[1], 3)}")
print(f"Kurtosis: {np.round(kurtosis(raw['Durchfluss'])[0], 3)}")
print(f"Kurtosis unbiased: {np.round(kurtosis(raw['Durchfluss'])[1], 3)}")

print("---------------------------------------------")

# quantiles
quantiles = [0.25, 0.5, 0.75]
quantiles_values = np.quantile(raw["Durchfluss"], quantiles)
print("Quantiles:")
for q, qv in zip(quantiles, quantiles_values):
    print(f"{q*100}%: {np.round(qv, 3)} m³/s")

# iqr
iqr_value = scipy.stats.iqr(raw["Durchfluss"])
print(f"IQR: {np.round(iqr_value, 3)} m³/s")
