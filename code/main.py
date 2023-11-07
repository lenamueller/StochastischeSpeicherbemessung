import numpy as np
import pandas as pd

from config import pegelname, report_path, image_path, fn_results

from utils.data_structures import read_data, check_path
from utils.primary_stats import sample_number, earliest_date, latest_date, max_val, min_val, \
    first_central_moment, second_central_moment, third_central_moment, fourth_central_moment, \
    standard_deviation_biased, standard_deviation_unbiased, skewness_biased, skewness_unbiased, \
    kurtosis_biased, kurtosis_unbiased, quartile, iqr
from utils.hydrological_values import hydro_values
from utils.consistency_check import missing_values, missing_dates, duplicates
from utils.trend_analysis import linreg, t_test_statistic, mk_test, detrend_signal
from utils.plotting import plot_raw, plot_hist, plot_trend, plot_components, plot_spectrum, plot_sin_waves, plot_saisonfigur
from utils.fft_analysis import calc_spectrum, get_dominant_frequency, season_signal



check_path(image_path)
check_path(report_path)


df = read_data(filename="Daten_Klingenthal_raw.txt")

info = pd.DataFrame(columns=["Name", "Wert", "Einheit"])

# -----------------------------------------
#           Primary information
# -----------------------------------------

print("Primärinformationen")
print("\tPegelname:", pegelname)
print("\tStichprobenumfang:", sample_number(df))
print("\tFrühestes Datum:", earliest_date(df))
print("\tSpätestes Datum:", latest_date(df))

# -----------------------------------------
#           Consistency check    
# -----------------------------------------

print("\nKonsistenzprüfung")
print("\tFehlwerte:", missing_values(df))
print("\tFehlende Zeitschritte:", missing_dates(df))
print("\tDuplikate:", duplicates(df))

# -----------------------------------------
#           Homogenity check
# -----------------------------------------

# TODO: #6 Check for outliers
# TODO: #7 double sum analysis

# -----------------------------------------
#           Stationarity check
# -----------------------------------------

# TODO: #8 Check for stationarity

# -----------------------------------------
#           Primary statistics
# -----------------------------------------

print("\nPrimärstatistik")
print("\tMinimum:", min_val(df))
print("\tMaximum:", max_val(df))
print("\t1. zentrales Moment:", first_central_moment(df))
print("\t2. zentrales Moment:", second_central_moment(df))
print("\t3. zentrales Moment:", third_central_moment(df))
print("\t4. zentrales Moment:", fourth_central_moment(df))
print("\tStandardabweichung (biased):", standard_deviation_biased(df))
print("\tStandardabweichung (unbiased):", standard_deviation_unbiased(df))
print("\tSkewness (biased):", skewness_biased(df))
print("\tSkewness (unbiased):", skewness_unbiased(df))
print("\tKurtosis (biased):", kurtosis_biased(df))
print("\tKurtosis (unbiased):", kurtosis_unbiased(df))
print("\t25%-Quantil:", quartile(df, which="Q1"))
print("\t50%-Quantil:", quartile(df, which="Q2"))
print("\t75%-Quantil:", quartile(df, which="Q3"))
print("\tInterquartilsabstand:", iqr(df))

hv = hydro_values(df)
for k, v in hv.items():
    print(f"\t{k}: {v}")
    
# -----------------------------------------
#               Distribution
# -----------------------------------------

# TODO: #9 Fit distribution to data
plot_raw(df)
plot_hist(df)

# -----------------------------------------
#           Trend analysis
# -----------------------------------------

# Statistical tests
print("\nTrendanalyse")
print("\tLineare Regression (Jahreswerte):", linreg(df, which="yearly"))
print("\tLineare Regression (Monatswerte):", linreg(df, which="monthly"))
print("\tTeststatistik lin. Regression (Jahreswerte):", np.round(t_test_statistic(df, which="yearly"), 3))
print("\tTeststatistik lin. Regression (Monatswerte):", np.round(t_test_statistic(df, which="monthly"), 3))
print("\tMK-Test (Jahreswerte):", mk_test(df, which="yearly"))
print("\tMK-Test (Monatswerte):", mk_test(df, which="monthly"))

# Data cleaning
df = detrend_signal(df)

# Plotting
plot_trend(df)

# -----------------------------------------
#           Seasonal analysis
# -----------------------------------------

print("\nSaisonanalyse")
freqs, spectrum = calc_spectrum(df)
freqs, period = get_dominant_frequency(freqs, spectrum, n=5)
print("\tTop 5 Frequenzen: ", freqs, "1/Monat")
print("\tTop 5 Periodendauern", period, "Monate")

# Data cleaning
df = season_signal(df)    

# Plotting
plot_spectrum(df)
plot_sin_waves(df)
plot_saisonfigur(df)

# -----------------------------------------
#        Autocorrelation analysis
# -----------------------------------------

# TODO: #12 Autocorrelation analysis

# Data cleaning
# TODO: #13 Create residual data

# Plotting
# TODO: plot all components

# -----------------------------------------
#               Save data
# -----------------------------------------
df.to_csv(fn_results, index=False)

plot_components(df)

