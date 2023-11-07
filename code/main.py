import numpy as np
import pandas as pd

from config import pegelname, report_path, image_path, fn_results

from utils.data_structures import read_data, check_path
import utils.statistics as st
from utils.consistency_check import missing_values, missing_dates, duplicates
from utils.plotting import plot_raw, plot_hist, plot_trend, plot_components, \
    plot_spectrum, plot_sin_waves, plot_saisonfigur, plot_acf


check_path(image_path)
check_path(report_path)


df = read_data(filename="Daten_Klingenthal_raw.txt")

info = pd.DataFrame(columns=["Name", "Wert", "Einheit"])

# -----------------------------------------
#           Primary information
# -----------------------------------------

print("Primärinformationen")
print("\tPegelname:", pegelname)
print("\tStichprobenumfang:", st.sample_number(df))
print("\tFrühestes Datum:", st.earliest_date(df))
print("\tSpätestes Datum:", st.latest_date(df))

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
for var in ["Durchfluss_m3s"]:
    print("\tMinimum:", st.min_val(df, var))
    print("\tMaximum:", st.max_val(df, var))
    print("\t1. zentrales Moment:", st.first_central_moment(df, var))
    print("\t2. zentrales Moment:", st.second_central_moment(df, var))
    print("\t3. zentrales Moment:", st.third_central_moment(df, var))
    print("\t4. zentrales Moment:", st.fourth_central_moment(df, var))
    print("\tStandardabweichung (biased):", st.standard_deviation_biased(df, var))
    print("\tStandardabweichung (unbiased):", st.standard_deviation_unbiased(df, var))
    print("\tSkewness (biased):", st.skewness_biased(df, var))
    print("\tSkewness (unbiased):", st.skewness_unbiased(df, var))
    print("\tKurtosis (biased):", st.kurtosis_biased(df, var))
    print("\tKurtosis (unbiased):", st.kurtosis_unbiased(df, var))
    print("\tQuantile (25%, 50%, 75%):", st.quartile(df, var))
    print("\tInterquartilsabstand:", st.iqr(df, var))

hv = st.hydro_values(df)
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
print("\tLineare Regression (Jahreswerte):", st.linreg(df, which="yearly"))
print("\tLineare Regression (Monatswerte):", st.linreg(df, which="monthly"))
print("\tTeststatistik lin. Regression (Jahreswerte):", np.round(st.t_test_statistic(df, which="yearly"), 3))
print("\tTeststatistik lin. Regression (Monatswerte):", np.round(st.t_test_statistic(df, which="monthly"), 3))
print("\tMK-Test (Jahreswerte):", st.mk_test(df, which="yearly"))
print("\tMK-Test (Monatswerte):", st.mk_test(df, which="monthly"))

# Data cleaning
df = st.detrend_signal(df)

# Plotting
plot_trend(df)

# -----------------------------------------
#           Seasonal analysis
# -----------------------------------------

print("\nSaisonanalyse")
freqs, spectrum = st.calc_spectrum(df)
freqs, period = st.get_dominant_frequency(freqs, spectrum, n=5)
print("\tTop 5 Frequenzen: ", freqs, "1/Monat")
print("\tTop 5 Periodendauern", period, "Monate")

# Data cleaning
df = st.season_signal(df)    

# Plotting
plot_spectrum(df)
plot_sin_waves(df)
plot_saisonfigur(df)

# -----------------------------------------
#        Autocorrelation analysis
# -----------------------------------------

print("\nAutokorrelationsanalyse")
print("\tMonatlich: ", st.monthly_autocorr(df, lag=1, var="saisonber"))
print("\tJährlich:", st.yearly_autocorr(df, lag=1, var="saisonber"))

# TODO: #12 Autocorrelation analysis

# Data cleaning
# TODO: #13 Create residual data

# Plotting
plot_acf(df)
plot_components(df)

# -----------------------------------------
#               Save data
# -----------------------------------------
df.to_csv(fn_results, index=False)



