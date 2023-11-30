import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import report_path, image_path, fn_results, pegelname

from utils.data_structures import read_data, check_path
import utils.statistics as st
from utils.consistency_check import missing_values, missing_dates, duplicates
from utils.plotting import plot_raw, plot_trend, plot_components, \
    plot_spectrum, plot_sin_waves, plot_characteristics, plot_acf, plot_dsk, \
    plot_breakpoint, pairplot, plot_thomasfiering
from utils.thomasfiering import parameter_xp, parameter_sp, parameter_rp, thomasfiering, fit_lognorm, _monthly_vals


check_path(image_path)
check_path(report_path)


df = read_data(f"data/others/Daten_{pegelname}.txt")

# -----------------------------------------
# Thomas Fiering model
# -----------------------------------------

# Parameter
tf_pars = pd.DataFrame()
tf_pars["Monat"] = np.arange(1, 13)
tf_pars["Mittelwert"] = [parameter_xp(df, i) for i in range(1, 13)]
tf_pars["Standardabweichung"] = [parameter_sp(df, i) for i in range(1, 13)]
tf_pars["Korrelationskoeffizient"] = [parameter_rp(df, i) for i in range(1, 13)]

tf_pars = tf_pars.round(4)
tf_pars.to_latex("reports/tomasfiering_parameters.tex", index=False)

print(tf_pars)

# Fitten
# import matplotlib.pyplot as plt
# from scipy.stats import lognorm
# x = np.arange(1, 13)
# bins = np.arange(0, 10, 0.5)
# for month in x:
#     df_month = df[df["Monat"].str.startswith(str(month).zfill(2))]
#     plt.hist(df_month["Durchfluss_m3s"].to_numpy(), bins=bins, label=month, alpha=0.3)
#     shape, loc, scale = fit_lognorm(df, month)
#     print("month", month, "parameters", shape, loc, scale)
#     plt.plot(x, lognorm.pdf(x, shape, loc, scale), label=month)

# plt.legend()
# plt.show()

# Zeitreihengenerierung
plot_thomasfiering(df, n=20)
    
exit()

# -----------------------------------------
#           Consistency check    
# -----------------------------------------

print("\nKonsistenzprüfung")
print("\tFehlwerte:", missing_values(df))
print("\tFehlende Zeitschritte:", missing_dates(df))
print("\tDuplikate:", duplicates(df))
print(
    "\tAusreißertest (IQR-Kriterium): ", 
    "obere Schranke:", st.outlier_test_iqr(df)[0],
    "untere Schranke:", st.outlier_test_iqr(df)[1],
    "Ausreißer:", st.outlier_test_iqr(df)[2]
    )
print(
    "\tAusreißertest (z-score): ", 
    "obere Schranke:", st.outlier_test_zscore(df)[0],
    "untere Schranke:", st.outlier_test_zscore(df)[1],
    "Ausreißer:", st.outlier_test_zscore(df)[2]
    )
print(
    "\tAusreißertest (Grubbs): ", 
    "Schranke:", st.outlier_test_grubbs(df)[0],
    "Ausreißer:", st.outlier_test_grubbs(df)[1]
    )

# -----------------------------------------
#           Homogenity check
# -----------------------------------------
print("\nHomogenitätsprüfung")
klingenthal = read_data("data/Daten_Klingenthal_raw.txt")
rothenthal = read_data("data/others/Daten_Rothenthal.txt")
kling, roth = st.double_sum(klingenthal["Durchfluss_m3s"], rothenthal["Durchfluss_m3s"])
plot_dsk(kling, roth)
print(
    "\tPettitt test:", st.pettitt_test(df),
    "Location:", df.iloc[st.pettitt_test(df).cp]   
    )
plot_breakpoint(df)

# -----------------------------------------
#           Stationarity check
# -----------------------------------------

print("\nStationaritätsprüfung")
print("\tADF-Test:", st.adf_test(df))

# -----------------------------------------
#           Trend analysis
# -----------------------------------------

print("\nTrendanalyse")
print("\tLineare Regression (Jahreswerte):", st.linreg(df, which="yearly"))
print("\tLineare Regression (Monatswerte):", st.linreg(df, which="monthly"))
print("\tTeststatistik lin. Regression (Jahreswerte):", np.round(st.t_test_statistic(df, which="yearly"), 3))
print("\tTeststatistik lin. Regression (Monatswerte):", np.round(st.t_test_statistic(df, which="monthly"), 3))
print("\tMK-Test (Jahreswerte):", st.mk_test(df, which="yearly"))
print("\tMK-Test (Monatswerte):", st.mk_test(df, which="monthly"))

plot_trend(df)

# -----------------------------------------
#      Seasonal and autocorr. analysis
# -----------------------------------------

print("\nSaisonanalyse")
freqs, spectrum = st.calc_spectrum(df)
freqs, period = st.get_dominant_frequency(freqs, spectrum, n=5)
print("\tTop 5 Frequenzen: ", freqs, "1/Monat")
print("\tTop 5 Periodendauern", period, "Monate")
plot_spectrum(df)
plot_sin_waves(df)


print("\nAutokorrelationsanalyse")
print("\tKonfidenzgrenzen: ", st.confidence_interval(df, lags=np.arange(0,24,1)))
st.calc_components(df)
print("\tJährlich:", st.yearly_autocorr(df, lag=1, var="saisonber"))
plot_acf(df, var="normiert")
plot_acf(df, var="Durchfluss_m3s")


plot_components(df)

# -----------------------------------------
#               Distribution
# -----------------------------------------

plot_raw(df)
plot_characteristics(df)
pairplot(df)
    

# -----------------------------------------
#               Statistics
# -----------------------------------------

names = ["Minimum", "Maximum", "1. zentrales Moment", "2. zentrales Moment", 
         "3. zentrales Moment", "4. zentrales Moment", "Standardabweichung (biased)",
         "Standardabweichung (unbiased)", "Skewness (biased)", "Skewness (unbiased)",
         "Kurtosis (biased)", "Kurtosis (unbiased)", "25%-Quantil", "50%-Quantil", 
         "75%-Quantil", "Interquartilsabstand", "Autokorrelation",
         "HHQ", "MHQ", "MQ", "MNQ", "NNQ"]

titles = ["Rohdaten", "Saisonbereinigt", "Zufall"]
vars = ["Durchfluss_m3s", "saisonber", "zufall"]
data = {"Name": names, "Rohdaten": [], "Saisonbereinigt": [], "Zufall": []}

for i in range(len(vars)):
    t = titles[i]
    data[titles[i]].append(st.min_val(df, vars[i])[0])
    data[titles[i]].append(st.max_val(df, vars[i])[0])
    data[titles[i]].append(st.first_central_moment(df, vars[i]))
    data[titles[i]].append(st.second_central_moment(df, vars[i]))
    data[titles[i]].append(st.third_central_moment(df, vars[i]))
    data[titles[i]].append(st.fourth_central_moment(df, vars[i]))
    data[titles[i]].append(st.standard_deviation_biased(df, vars[i]))
    data[titles[i]].append(st.standard_deviation_unbiased(df, vars[i]))
    data[titles[i]].append(st.skewness_biased(df, vars[i]))
    data[titles[i]].append(st.skewness_unbiased(df, vars[i]))
    data[titles[i]].append(st.kurtosis_biased(df, vars[i]))
    data[titles[i]].append(st.kurtosis_unbiased(df, vars[i]))
    data[titles[i]].append(st.quantiles(df, 0.25, vars[i]))
    data[titles[i]].append(st.quantiles(df, 0.50, vars[i]))
    data[titles[i]].append(st.quantiles(df, 0.75, vars[i]))
    data[titles[i]].append(st.iqr(df, vars[i]))
    data[titles[i]].append(st.autocorrelation(df, vars[i]))
    hydro_vals = st.hydro_values(df)
    if i == 0:
        data[titles[i]].append(hydro_vals["HHQ"][0])
        data[titles[i]].append(hydro_vals["MHQ"])
        data[titles[i]].append(hydro_vals["MQ"])
        data[titles[i]].append(hydro_vals["MNQ"])
        data[titles[i]].append(hydro_vals["NNQ"][0])
    else:
        data[titles[i]].append("-")
        data[titles[i]].append("-")
        data[titles[i]].append("-")
        data[titles[i]].append("-")
        data[titles[i]].append("-")
        
    
    
df_statistics = pd.DataFrame.from_dict(data)
df_statistics.round({"Rohdaten":3, "Saisonbereinigt":3, "Zufall":3})
df_statistics.to_latex("reports/statistics.tex", index=False)
df_statistics.to_csv("reports/statistics.csv", index=False)

# -----------------------------------------
#               Save data
# -----------------------------------------

df.to_csv(fn_results, index=False)

