import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(1, '/home/lena/Dokumente/FGB/StochastischeSpeicherbemessung/')

from settings import PEGEL, N_GEN_TIMESERIES
from utils.read import read_raw_data
from plotting_func import plot_thomasfierung_eval, plot_monthly_fitting
from thomas_fiering import gen_timeseries, parameter_rp, parameter_sp, parameter_xp


paths = [
    "2_Zeitreihengenerierung/results/",
    f"2_Zeitreihengenerierung/results/{PEGEL}",
    f"2_Zeitreihengenerierung/results/{PEGEL}/plots"
    ]

for path in paths:
    if not os.path.exists(path):
        os.mkdir(path)

# ---------------------------------------
# Rohdaten einlesen
# ---------------------------------------

df = read_raw_data(f"Rohdaten/Daten_{PEGEL}.txt")

# ---------------------------------------
# Modellparametrisierung
# ---------------------------------------

parameters = pd.DataFrame()
month_list = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parameters["Monat"] = month_list
parameters["Parameter xp"] = [parameter_xp(df, i) for i in month_list]
parameters["Parameter sp"] = [parameter_sp(df, i) for i in month_list]
parameters["Parameter rp"] = [parameter_rp(df, i) for i in month_list]
parameters.to_csv(f"2_Zeitreihengenerierung/results/{PEGEL}/ThomasFiering_Parameter.csv", index=False)
    
# ---------------------------------------        
# Generierung von Zeitreihen
# ---------------------------------------

gen_data = pd.DataFrame()
for i in range(N_GEN_TIMESERIES):
    print("Generiere Zeitreihe: ", i+1)
    gen_data[f"G{str(i+1).zfill(3)}"] = gen_timeseries(df)
gen_data.to_csv(f"2_Zeitreihengenerierung/results/{PEGEL}/GenerierteZeitreihen.csv", index=False)

# ---------------------------------------
# Plots
# ---------------------------------------

raw_data = read_raw_data(f"Rohdaten/Daten_{PEGEL}.txt")
gen_data["Monat"] = np.tile(["11","12","01","02","03","04","05","06","07","08","09","10"], 80)

plot_monthly_fitting(
    df=raw_data,
    fn=f"2_Zeitreihengenerierung/results/{PEGEL}/plots/AnpassungGammaLogNV.png"
    )

plot_thomasfierung_eval(
    raw_data=raw_data, gen_data=gen_data,
    fn=f"2_Zeitreihengenerierung/results/{PEGEL}/plots/ThomasFiering_Evaluation.png"
)
