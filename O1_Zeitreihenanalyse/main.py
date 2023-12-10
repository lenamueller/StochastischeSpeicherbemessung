import numpy as np
import pandas as pd
import logging
import os
import sys
sys.path.insert(1, '/home/lena/Documents/Studium/Master_Hydrologie/2_Flussgebietsbewirtschaftung/StochastischeSpeicherbemessung/')
import warnings
warnings.filterwarnings("ignore")

from settings import PEGEL_NAMES, REF_PEGEL
from utils.read import read_raw_data
from utils.stats import *

from check_consistency import *
from check_homogenity import *
from check_stationarity import *
from comp_autocorr import *
from comp_seasonal import *
from comp_trend import *
from plotting_func import *


for PEGEL in PEGEL_NAMES:

    paths = [
        "O1_Zeitreihenanalyse/results/",
        f"O1_Zeitreihenanalyse/results/{PEGEL}",
        f"O1_Zeitreihenanalyse/results/{PEGEL}/plots"
        ]

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
    
    plot_path = f"O1_Zeitreihenanalyse/results/{PEGEL}/plots/"

    log_path = f"O1_Zeitreihenanalyse/results/Zeitreihenanalyse.log"
    logging.basicConfig(filename=log_path, 
                        format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S',
                        level=logging.INFO, filemode='w')
    
    logging.info("-----------------------------------------")
    logging.info(f"PEGEL: {PEGEL}")
    logging.info("-----------------------------------------")
    
    # -----------------------------------------
    # Einlesen der Rohdaten
    # -----------------------------------------
    
    df = read_raw_data(f"Rohdaten/Daten_{PEGEL}.txt")
    df["Datum"] = pd.to_datetime(df["Monat"], format="%m/%Y")

    plot_raw(df, fn=plot_path + "Rohdaten.png")
    
    # -----------------------------------------
    # Check: Konsistenz
    # -----------------------------------------

    logging.info("-----------------------------------------")
    logging.info("Check: Konsistenz")
    logging.info("-----------------------------------------")

    logging.info(f"Fehlwerte: {missing_values(df)}")
    logging.info(f"Fehlende Zeitschritte: {missing_dates(df)}")
    logging.info(f"Duplikate: {duplicates(df)}")
    logging.info(f"Ausreißertest (IQR): {outlier_test_iqr(df)}")
    logging.info(f"Ausreißertest (z-score): {outlier_test_zscore(df)}")
    logging.info(f"Ausreißertest (Grubbs): {outlier_test_grubbs(df)}")

    # -----------------------------------------
    # Check: Homogenität
    # -----------------------------------------
    
    logging.info("-----------------------------------------")
    logging.info("Check: Homogenität")
    logging.info("-----------------------------------------")
    
    # Doppelsummenanalyse
    test, ref = double_sum(
        test_gauge=df["Durchfluss_m3s"],
        ref_gauge=read_raw_data(f"Rohdaten/Daten_{REF_PEGEL[PEGEL]}.txt")["Durchfluss_m3s"])

    plot_dsk(test, ref, fn=plot_path + "Doppelsummenanalyse.png")
    
    # Bruckpunktanalyse (Pettitt test)
    res = pettitt_test(df)
    bp = df["Monat"].iloc[pettitt_test(df).cp]
    logging.info(f"Pettitt test: {res}, Bruchpunkt: {bp}")
    
    plot_breakpoint(df, res, fn=plot_path + "Bruchpunktanalyse.png")

    # -----------------------------------------
    # Check: Stationarität
    # -----------------------------------------
    
    logging.info("-----------------------------------------")
    logging.info("Check: Stationarität")
    logging.info("-----------------------------------------")
    
    logging.info(f"ADF-Test: {adf_test(df)}")
    
    # -----------------------------------------
    # Analyse: Trendkomponente
    # -----------------------------------------

    logging.info("-----------------------------------------")
    logging.info("Analyse: Trendkomponente")
    logging.info("-----------------------------------------")
    
    # Lin. Regression / t-Test
    linreg_m = linreg(df, which="monthly")
    linreg_y = linreg(df, which="yearly")
    logging.info(f"Lineare Regression (Jahreswerte): {linreg_y}")
    logging.info(f"Lineare Regression (Monatswerte): {linreg_m}")
    logging.info(f"""Teststatistik lin. Regression (Jahreswerte): {np.round(t_test_statistic(df, which="yearly"), 3)}""")
    logging.info(f"""Teststatistik lin. Regression (Monatswerte): {np.round(t_test_statistic(df, which="monthly"), 3)}""")
    
    # Mann-Kendall-Test
    mk_m = mk.original_test(df["Durchfluss_m3s"], alpha=0.05)
    mk_y = mk.seasonal_test(df["Durchfluss_m3s"], alpha=0.05, period=12)
    logging.info(f"MK-Test (Jahreswerte): {mk_y}")
    logging.info(f"MK-Test (Monatswerte): {mk_m}")
    
    # Moving average
    ma_m = moving_average(df, which="monthly", window=12)
    ma_y = moving_average(df, which="yearly", window=5)
    
    plot_trend(df, 
               linreg_m=linreg_m, linreg_y=linreg_y, 
               mk_m=mk_m, mk_y=mk_y, 
               ma_m=ma_m, ma_y=ma_y,
               fn=plot_path + "Trendanalyse.png"
               )

    # Berechnung Zeitreihenkomponenten
    df["trendber"] = np.mean(df["Durchfluss_m3s"].to_numpy()) + \
        scipy.signal.detrend(df["Durchfluss_m3s"].to_numpy(), type="linear")
    df["trend"] = df["Durchfluss_m3s"].to_numpy() - df["trendber"].to_numpy()

    # -----------------------------------------
    # Analyse: Saisonkomponente
    # -----------------------------------------
    
    logging.info("-----------------------------------------")
    logging.info("Analyse: Saisonkomponente")
    logging.info("-----------------------------------------")
    
    freqs, spectrum = calc_spectrum(df)

    plot_spectrum(freqs, spectrum, 
                  fn=plot_path + "Spektralanalyse.png")
    
    freqs, period = get_dominant_frequency(freqs, spectrum, n=5)
    logging.info(f"Top 5 Frequenzen: {freqs} 1/Monat")
    logging.info(f"Top 5 Periodendauern {period} Monate")

    plot_sin_waves(freqs, period, 
                   fn=plot_path + "DominierendeFrequenzen.png")
    
    # Berechnung Zeitreihenkomponenten
    df["saisonfigur_mean"] = np.tile(binned_stats(df, var="Durchfluss_m3s", bin="monthly", func=np.mean), 40)
    df["saisonfigur_std"] = np.tile(binned_stats(df, var="Durchfluss_m3s", bin="monthly", func=np.std), 40)
    df["saisonber"] = df["Durchfluss_m3s"] - df["saisonfigur_mean"]
    df["normiert"] = (df["Durchfluss_m3s"] - df["saisonfigur_mean"]) / df["saisonfigur_std"]

    # -----------------------------------------
    # Analyse: Autokorrelative Komponente
    # -----------------------------------------
    
    logging.info("-----------------------------------------")
    logging.info("Analyse: Autokorrelative Komponente")
    logging.info("-----------------------------------------")

    lags = np.arange(0,51,1)
    lower_conf, upper_conf = confidence_interval(df, lags)
    ac_normiert = [df["normiert"].autocorr(lag=i) for i in lags]
    ac_raw = [df["Durchfluss_m3s"].autocorr(lag=i) for i in lags]
    
    pd.DataFrame(data={
        "Lags": lags,
        "Autokorrelation_normierteDaten": ac_normiert,
        "Autokorrelation_Rohdaten": ac_raw,
        "UnterKonfGrenze": lower_conf,
        "ObereKonfGrenze": upper_conf
        }).to_csv(f"O1_Zeitreihenanalyse/results/{PEGEL}/Autokorrelation.csv", index=False)
    
    plot_acf(lags, ac_raw, lower_conf=lower_conf, upper_conf=upper_conf, 
             fn=plot_path + "Autokorrelation_Rohdaten.png")
    plot_acf(lags, ac_normiert,lower_conf=lower_conf, upper_conf=upper_conf, 
             fn=plot_path + "Autokorrelation_normierteDaten.png")

    # Berechnung Zeitreihenkomponenten
    df["autokorr_saisonfigur"] = np.tile(monthly_autocorr(df=df), 40)
    df["autokorr"] = df["autokorr_saisonfigur"] * df["normiert"]

    # -----------------------------------------
    # Analyse: Irreguläre Komponente
    # -----------------------------------------
    
    logging.info("-----------------------------------------")
    logging.info("Analyse: Irreguläre Komponente")
    logging.info("-----------------------------------------")
    
    # Berechnung Zeitreihenkomponenten
    df["zufall"] = df["normiert"] - df["autokorr"]

    # plot comparison of components    
    plot_characteristics_short(df, fn=plot_path + "Zeitreihenkomponenten_Vergleich.png")
    pairplot(df, fn=plot_path + "Zeitreihenkomponenten_Pairplot.png")
    plot_components(df, fn=plot_path + "Zeitreihenkomponenten.png")

    # -----------------------------------------
    # Statistiken
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
        data[titles[i]].append(min_val(df, vars[i])[0])
        data[titles[i]].append(max_val(df, vars[i])[0])
        data[titles[i]].append(central_moment(df, nth=1, var=vars[i]))
        data[titles[i]].append(central_moment(df, nth=2, var=vars[i]))
        data[titles[i]].append(central_moment(df, nth=3, var=vars[i]))
        data[titles[i]].append(central_moment(df, nth=4, var=vars[i]))
        data[titles[i]].append(standard_deviation(df, bias=True, var=vars[i]))
        data[titles[i]].append(standard_deviation(df, bias=False, var=vars[i]))
        data[titles[i]].append(skewness(df, bias=True, var=vars[i]))
        data[titles[i]].append(skewness(df, bias=False, var=vars[i]))
        data[titles[i]].append(kurtosis(df, bias=True, var=vars[i]))
        data[titles[i]].append(kurtosis(df, bias=False, var=vars[i]))
        data[titles[i]].append(quantiles(df, 0.25, vars[i]))
        data[titles[i]].append(quantiles(df, 0.50, vars[i]))
        data[titles[i]].append(quantiles(df, 0.75, vars[i]))
        data[titles[i]].append(iqr(df, vars[i]))
        data[titles[i]].append(autocorrelation(df, vars[i]))
        hydro_vals = hydro_values(df)
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
    df_statistics.to_csv(f"O1_Zeitreihenanalyse/results/{PEGEL}/Statistiken.csv", index=False)
    
    # -----------------------------------------
    # Speichern der Zeitreihenkomponenten
    # -----------------------------------------
    
    df.rename(columns=var_remapper, inplace=True)
    df.drop(columns=["Datum"], inplace=True)
    df.to_csv(f"O1_Zeitreihenanalyse/results/{PEGEL}/Zeitreihenkomponenten.csv", index=False)
    
    logging.info("-----------------------------------------")
    logging.info("Ende")
    logging.info("-----------------------------------------")