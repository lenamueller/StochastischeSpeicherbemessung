import matplotlib.pyplot as plt
import pymannkendall as mk
import numpy as np
import pandas as pd

from config import pegelname, report_path, image_path

from utils.data_structures import read_data, check_path
from utils.binned_stats import mean
from utils.primary_stats import sample_number, earliest_date, latest_date, max_val, min_val, \
    first_central_moment, second_central_moment, third_central_moment, fourth_central_moment, \
    standard_deviation_biased, standard_deviation_unbiased, skewness_biased, skewness_unbiased, \
    kurtosis_biased, kurtosis_unbiased, quartile, iqr
from utils.hydrological_values import hydro_values
from utils.consistency_check import missing_values, missing_dates, duplicates
from utils.trend_analysis import linreg, test_statistic, detrend_signal
from utils.plotting import plot_raw, plot_hist, plot_trend, plot_detrending, plot_spectrum, plot_sin_waves, plot_saisonfigur
from utils.fft_analysis import calc_spectrum, get_dominant_frequency


def agenda(df: pd.DataFrame):
    check_path(image_path)
    check_path(report_path)

    plot_raw(df)
    plot_hist(df)
    
    info = pd.DataFrame(columns=["Name", "Wert", "Einheit"])
    
    # Primary information
    info.loc[len(info)] = ["Pegelname", pegelname, "-"]
    info.loc[len(info)] = ["Stichprobenumfang", sample_number(df), "-"]
    info.loc[len(info)] = ["Frühestes Datum", earliest_date(df), "-"]
    info.loc[len(info)] = ["Spätestes Datum", latest_date(df), "-"]
    
    # Consistency check    
    info.loc[len(info)] = ["Fehlwerte", missing_values(df), "-"]
    info.loc[len(info)] = ["Fehlende Zeitschritte", missing_dates(df), "-"]
    info.loc[len(info)] = ["Duplikate", duplicates(df), "m³/s"]
    
    # Homogenity check
    # TODO: #6 Check for outliers
    # TODO: #7 double sum analysis
    
    # Stationarity check
    # TODO: #8 Check for stationarity
    
    # Primary stats
    info.loc[len(info)] = ["Minimum", np.round(min_val(df), 3), "m³/s"]
    info.loc[len(info)] = ["Maximum", np.round(max_val(df), 3), "m³/s"]
    info.loc[len(info)] = ["1. zentrales Moment", np.round(first_central_moment(df), 3), "m³/s"]
    info.loc[len(info)] = ["2. zentrales Moment", np.round(second_central_moment(df), 3), "(m³/s)²"]
    info.loc[len(info)] = ["3. zentrales Moment", np.round(third_central_moment(df), 3), "(m³/s)³"]
    info.loc[len(info)] = ["4. zentrales Moment", np.round(fourth_central_moment(df), 3), "(m³/s)⁴"]
    info.loc[len(info)] = ["Standardabweichung (biased)", np.round(standard_deviation_biased(df), 3), "m³/s"]
    info.loc[len(info)] = ["Standardabweichung (unbiased)", np.round(standard_deviation_unbiased(df), 3), "m³/s"]
    info.loc[len(info)] = ["Skewness (biased)", np.round(skewness_biased(df), 3), "-"]
    info.loc[len(info)] = ["Skewness (unbiased)", np.round(skewness_unbiased(df), 3), "-"]
    info.loc[len(info)] = ["Kurtosis (biased)", np.round(kurtosis_biased(df), 3), "-"]
    info.loc[len(info)] = ["Kurtosis (unbiased)", np.round(kurtosis_unbiased(df), 3), "-"]
    info.loc[len(info)] = ["25%-Quantil", np.round(quartile(df, which="Q1"), 3), "m³/s"]
    info.loc[len(info)] = ["50%-Quantil", np.round(quartile(df, which="Q2"), 3), "m³/s"]
    info.loc[len(info)] = ["75%-Quantil", np.round(quartile(df, which="Q3"), 3), "m³/s"]
    info.loc[len(info)] = ["Interquartilsabstand", np.round(iqr(df), 3), "m³/s"]
    
    # Distribution
    # TODO: #9 Fit distribution to data

    # Hydrological values
    hv = hydro_values(df)
    for k, v in hv.items():
        info.loc[len(info)] = [k, v, "m³/s"]
        
    # Trend analysis: linear regression and t-test
    info.loc[len(info)] = ["Lineare Regression (Jahreswerte)", linreg(df, which="yearly"), "-"]
    info.loc[len(info)] = ["Lineare Regression (Monatswerte)", linreg(df, which="monthly"), "-"]
    info.loc[len(info)] = ["Teststatistik lin. Regression (Jahreswerte)", np.round(test_statistic(df, which="yearly"), 3), "-"]
    info.loc[len(info)] = ["Teststatistik lin. Regression (Monatswerte)", np.round(test_statistic(df, which="monthly"), 3), "-"]
    
    # Trend analysis: Theil-Sen regression and MK-test
    info.loc[len(info)] = ["MK-Test (Jahreswerte)", mk.original_test(mean(df, which="yearly"), alpha=0.05), "-"]
    info.loc[len(info)] = ["MK-Test (Monatswerte)", mk.seasonal_test(df["Durchfluss_m3s"].to_numpy(), alpha=0.05, period=12), "-"]
    
    # Detrending
    df_detrended = detrend_signal(df)
    df_detrended.to_csv(f"data/{pegelname}_detrended.csv", index=False)
    
    plot_trend(df)
    plot_detrending(df, df_detrended)
    
    # Seasonal analysis
    freqs, spectrum = calc_spectrum(df)
    freqs, period = get_dominant_frequency(freqs, spectrum, n=5)
    info.loc[len(info)] = ["5 dominantesten Frequenzen", freqs, "1/Monat"]
    info.loc[len(info)] = ["5 dominantesten Periodendauern", period, "Monate"]
    
    # TODO: #11 Create seasonal data
    
    plot_spectrum(df)
    plot_sin_waves(df)
    plot_saisonfigur(df)
    
    # Autocorrelation analysis
    # TODO: #12 Autocorrelation analysis
    # TODO: #13 Create residual data
    

    info.to_csv(f"reports/{pegelname}_TSA.csv", index=False)

    
print("Done!")
    
        
fns = [
    f"Daten_{pegelname}_raw.txt",
    # f"Daten_{pegelname}_detrended.txt",
    # f"Daten_{pegelname}_seasonal.txt",
    # f"Daten_{pegelname}_residual.txt"
]

for fn in fns:
    df = read_data(filename=fn)
    df.to_csv(f"data/{pegelname}_raw.csv", index=False)
    agenda(df)