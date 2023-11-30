import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import report_path, image_path, fn_results, pegelname

from utils.data_structures import read_data, check_path, _monthly_vals
import utils.statistics as st
from utils.consistency_check import missing_values, missing_dates, duplicates
from utils.plotting import plot_raw, plot_trend, plot_components, \
    plot_spectrum, plot_sin_waves, plot_characteristics, plot_acf, plot_dsk, \
    plot_breakpoint, pairplot, plot_thomasfiering, plot_monthly_fitting,  \
    plot_thomasfierung_eval
from utils.thomasfiering import parameter_xp, parameter_sp, parameter_rp, thomasfiering


check_path(image_path)
check_path(report_path)


df = read_data(f"data/others/Daten_{pegelname}.txt")

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
# Thomas Fiering model
# -----------------------------------------

# fit model
tf_pars = pd.DataFrame()
tf_pars["Monat"] = np.arange(1, 13)
tf_pars["Mittelwert"] = [parameter_xp(df, i) for i in range(1, 13)]
tf_pars["Standardabweichung"] = [parameter_sp(df, i) for i in range(1, 13)]
tf_pars["Korrelationskoeffizient"] = [parameter_rp(df, i) for i in range(1, 13)]

tf_pars = tf_pars.round(4)
tf_pars.to_csv(f"data/{pegelname}_tomasfiering_parameters.csv", index=False)
tf_pars.to_latex(f"data/{pegelname}_tomasfiering_parameters.tex", index=False)

# check distribution
plot_monthly_fitting(df)

# generate time series
n = 100
gen_data = pd.DataFrame(
    data=[thomasfiering(df) for _ in range(n)],
    index=np.arange(1, n+1), 
    columns=[11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

gen_data.to_csv(f"data/{pegelname}_thomasfiering_timeseries.csv", index=True)
gen_data.iloc[:10].round(3).to_latex(f"data/{pegelname}_thomasfiering_timeseries_first10.tex", index=True)
plot_thomasfiering(df, gen_data.to_numpy(), n=100)
plot_thomasfierung_eval(df, gen_data.to_numpy())

# -----------------------------------------
#               Statistics
# -----------------------------------------

names = ["Minimum", "Maximum", "1. zentrales Moment", "2. zentrales Moment", 
         "3. zentrales Moment", "4. zentrales Moment", "Standardabweichung (biased)",
         "Standardabweichung (unbiased)", "Skewness (biased)", "Skewness (unbiased)",
         "Kurtosis (biased)", "Kurtosis (unbiased)", "25%-Quantil", "50%-Quantil", 
         "75%-Quantil", "Interquartilsabstand", "Autokorrelation",
         "HHQ", "MHQ", "MQ", "MNQ", "NNQ"]

titles = ["Rohdaten", "Saisonbereinigte Zeitreihe", "Zufallskomponente der Zeitreihe"]
vars = ["Durchfluss_m3s", "saisonber", "zufall"]
data = {"Name": names, "Rohdaten": [], "Saisonbereinigte Zeitreihe": [], "Zufallskomponente der Zeitreihe": []}

for i in range(len(vars)):
    t = titles[i]
    vars[i]
    data[titles[i]].append(st.min_val(df, vars[i])[0])
    data[titles[i]].append(st.max_val(df, vars[i])[0])
    data[titles[i]].append(st.central_moment(df, nth=1, var=vars[i]))
    data[titles[i]].append(st.central_moment(df, nth=2, var=vars[i]))
    data[titles[i]].append(st.central_moment(df, nth=3, var=vars[i]))
    data[titles[i]].append(st.central_moment(df, nth=4, var=vars[i]))
    data[titles[i]].append(st.standard_deviation(df, bias=True, var=vars[i]))
    data[titles[i]].append(st.standard_deviation(df, bias=False, var=vars[i]))
    data[titles[i]].append(st.skewness(df, bias=True, var=vars[i]))
    data[titles[i]].append(st.skewness(df, bias=False, var=vars[i]))
    data[titles[i]].append(st.kurtosis(df, bias=True, var=vars[i]))
    data[titles[i]].append(st.kurtosis(df, bias=False, var=vars[i]))
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
df_statistics.round({"Rohdaten":3, "Saisonbereinigte Zeitreihe":3, "Zufallskomponente der Zeitreihe":3})
df_statistics.to_latex("reports/statistics.tex", index=False)
df_statistics.to_csv("reports/statistics.csv", index=False)

# -----------------------------------------
#               Save data
# -----------------------------------------

df.to_csv(fn_results, index=False)

