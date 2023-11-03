import sys
import time
import numpy as np

from setup import tu_mediumblue, tu_red, tu_grey
from Timeseries import TimeSeries


# read first argument from command line
pegelname = sys.argv[1]

t = TimeSeries(pegelname)




print("------------------------------------------------------------")
print("PRIMÄRE INFORMATION")
print("\tPegelname:", t.name)
print("\tStichprobenumfang:", t.sample_number())
print("\tFrühestes Datum:", t.earliest_date())
print("\tSpätestes Datum:", t.latest_date())

print("------------------------------------------------------------")
print("PRIMÄRSTATISTIK")
print(f"\tMinimum: {np.round(t.min_val(), 3)}")
print(f"\tMaximum: {np.round(t.max_val(), 3)}")
print(f"\t1. zentrales Moment: {np.round(t.first_central_moment(), 3)}")
print(f"\t2. zentrales Moment: {np.round(t.second_central_moment(), 3)}")
print(f"\t3. zentrales Moment: {np.round(t.third_central_moment(), 3)}")
print(f"\t4. zentrales Moment: {np.round(t.fourth_central_moment(), 3)}")
print(f"\tSkewness: {np.round(t.skewness_biased(), 3)}")
print(f"\tSkewness unbiased: {np.round(t.skewness_unbiased(), 3)}")
print(f"\tKurtosis: {np.round(t.kurtois_unbiased(), 3)}")
print(f"\tKurtosis unbiased: {np.round(t.kurtois_unbiased(), 3)}")
print(f"\t25%-Quartil {np.round(t.quartile(0.25), 3)}")
print(f"\t50%-Quartil {np.round(t.quartile(0.50), 3)}")
print(f"\t75%-Quartil {np.round(t.quartile(0.75), 3)}")
print(f"\tIQR", t.iqr())

print("------------------------------------------------------------")
print("HYDROLOGISCHE KENNWERTE")
hv = t.hydro_values()
for k, v in hv.items():
    print(f"\t{k}: {v}")

print("------------------------------------------------------------")
print("KONSISTENZ - CHECK")
print("\tFehlwerte:", t.missing_values())
print("\tFehlende Zeitschritte:", t.missing_dates())
print("\tDuplikate:", t.duplicates())

print("------------------------------------------------------------")
print("FFT")
t.plot_spectrum()
t.plot_sin_waves()
t.plot_saisonfigur()

print("------------------------------------------------------------")
print("PLOTTEN ... ")
t.plot_raw()
t.plot_hist()

print("------------------------------------------------------------")
print("FERTIG!")
print("Created:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))