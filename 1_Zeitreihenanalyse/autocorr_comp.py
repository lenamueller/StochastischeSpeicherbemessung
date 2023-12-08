import numpy as np
import pandas as pd

from config import ALPHA, pegelname
from utils.plotting import plot_acf
from utils.statistics import monthly_autocorr


def confidence_interval(df: pd.DataFrame, lags: list[float]) -> tuple[list[float], list[float]]:
    """Returns the confidence interval."""
    k = lags
    n = len(df)
    T_ALPHA = 1.645 # alpha = 0.05
    lower_conf = (-1 - T_ALPHA*np.sqrt(n-k-1)) / (n-k+1)
    upper_conf = (1 + T_ALPHA*np.sqrt(n-k-1)) / (n-k+1)
    return lower_conf, upper_conf

def autocorr_comp(df: pd.DataFrame):
    print("\n--------------------------------------")
    print("\nBestimmung der autokorrelativen Komponente\n")
    
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
        }).to_csv(f"data/{pegelname}_autokorr.csv", index=False)
    
    print(f"-> data/{pegelname}_autokorr.csv")    
    
    plot_acf(lags, ac_raw, lower_conf=lower_conf, upper_conf=upper_conf, fn_extension="raw")
    plot_acf(lags, ac_normiert,lower_conf=lower_conf, upper_conf=upper_conf, fn_extension="normiert")

    # calculate components
    df["autokorr_saisonfigur"] = np.tile(monthly_autocorr(df=df), 40)
    df["autokorr"] = df["autokorr_saisonfigur"] * df["normiert"]
