import sys
import time
import numpy as np

from utils.data_structures import read_data   
from utils.primary_stats import sample_number, earliest_date, latest_date, max_val, min_val, quartile, iqr
from utils.primary_stats import first_central_moment, second_central_moment, third_central_moment, fourth_central_moment
from utils.primary_stats import skewness_biased, skewness_unbiased, kurtois_biased, kurtois_unbiased
from utils.hydrological_values import hydro_values
from utils.consistency_check import missing_values, missing_dates, duplicates
from utils.trend_analysis import linreg_monthly, linreg_yearly
from utils.plotting import plot_raw, plot_hist, plot_trend, plot_spectrum, plot_sin_waves, plot_saisonfigur


# read first argument from command line
pegelname = sys.argv[1]
df = read_data(filename=f"Daten_{pegelname}.txt")

# todo: create beautiful table for output or pandas.DataFrame

print("------------------------------------------------------------")
print("PRIMÄRE INFORMATION")
print("\tPegelname:", pegelname)
print("\tStichprobenumfang:", sample_number(df))
print("\tFrühestes Datum:", earliest_date(df))
print("\tSpätestes Datum:", latest_date(df))

print("------------------------------------------------------------")
print("PRIMÄRSTATISTIK")
print(f"\tMinimum: {np.round(min_val(df), 3)}")
print(f"\tMaximum: {np.round(max_val(df), 3)}")
print(f"\t1. zentrales Moment: {np.round(first_central_moment(df), 3)}")
print(f"\t2. zentrales Moment: {np.round(second_central_moment(df), 3)}")
print(f"\t3. zentrales Moment: {np.round(third_central_moment(df), 3)}")
print(f"\t4. zentrales Moment: {np.round(fourth_central_moment(df), 3)}")
print(f"\tSkewness: {np.round(skewness_biased(df), 3)}")
print(f"\tSkewness unbiased: {np.round(skewness_unbiased(df), 3)}")
print(f"\tKurtosis: {np.round(kurtois_unbiased(df), 3)}")
print(f"\tKurtosis unbiased: {np.round(kurtois_unbiased(df), 3)}")
print(f"\t25%-Quartil {np.round(quartile(df, 0.25), 3)}")
print(f"\t50%-Quartil {np.round(quartile(df, 0.50), 3)}")
print(f"\t75%-Quartil {np.round(quartile(df, 0.75), 3)}")
print(f"\tIQR", iqr(df))

print("------------------------------------------------------------")
print("HYDROLOGISCHE KENNWERTE")
hv = hydro_values()
for k, v in hv.items():
    print(f"\t{k}: {v}")

print("------------------------------------------------------------")
print("KONSISTENZ - CHECK")
print("\tFehlwerte:", missing_values(df))
print("\tFehlende Zeitschritte:", missing_dates(df))
print("\tDuplikate:", duplicates(df))

print("------------------------------------------------------------")
print("TREND")
print("\tLineare Regression (Jahreswerte):", linreg_yearly(df))
print("\tLineare Regression (Monatswerte):", linreg_monthly(df))

print("------------------------------------------------------------")
print("FFT")


print("------------------------------------------------------------")
print("PLOTTEN ... ")
plot_raw(df)
plot_hist(df)
plot_trend(df)
plot_spectrum(df)
plot_sin_waves()
plot_saisonfigur(df)

print("------------------------------------------------------------")
print("FERTIG!")
print("Created:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))