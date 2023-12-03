import numpy as np
import pandas as pd
import scipy

from config import ALPHA, pegelname
from utils.plotting import plot_acf


def autocorrelation(df: pd.DataFrame, var: str, lag: int = 1) -> float:
    """Returns the autocorrelation function."""
    return pd.Series(df[var]).autocorr(lag=lag)

def confidence_interval(df: pd.DataFrame, lags: list[float]) -> tuple[list[float], list[float]]:
    """Returns the confidence interval."""
    k = lags
    n = len(df)
    T_ALPHA = 1.645 # alpha = 0.05
    lower_conf = (-1 - T_ALPHA*np.sqrt(n-k-1)) / (n-k+1)
    upper_conf = (1 + T_ALPHA*np.sqrt(n-k-1)) / (n-k+1)
    return lower_conf, upper_conf

def monthly_autocorr(df: pd.DataFrame, var: str = "saisonber", which: str = "maniak") -> list[float]:
    """Returns a list of monthly autocorrelations for lag (k) = 1."""

    months = df["Monat"]
    pairs = [("11", "12"), ("12", "01"), ("01", "02"), ("02", "03"),
             ("03", "04"), ("04", "05"), ("05", "06"), ("06", "07"),
             ("07", "08"), ("08", "09"), ("09", "10"), ("10", "11")]
    
    coeff = []
    for i in range(12):
        first_month, second_month = pairs[i]
        x_i = df[var][months.str.startswith(first_month)].tolist()
        x_ik = df[var][months.str.startswith(second_month)].tolist()
        assert len(x_i) == len(x_ik)
        
        mean_x_i = np.mean(x_i)
        mean_x_ik = np.mean(x_ik)
        std_x_i = np.std(x_i)
        std_x_ik = np.std(x_ik)
        k = 1
        n = len(x_i)
        
        if which == "pearson":
            coeff.append(scipy.stats.stats.pearsonr(x_i, x_ik).statistic)
        elif which == "maniak":
            prod = [(i - mean_x_i) * (ik - mean_x_ik) for i, ik in zip(x_i, x_ik)]
            r_k_maniak = (sum(prod[:-k])) / (std_x_i*std_x_ik) / (n-k)
            coeff.append(r_k_maniak)
        else:
            raise ValueError("which must be 'pearson' or 'maniak'")
    return coeff

def yearly_autocorr(
    df: pd.DataFrame,
    lag: int,
    var: str = "Durchfluss_m3s") -> list[float]:
    """Returns a list of yearly autocorrelations."""
    arr = np.reshape(df[var].to_numpy(), (-1, 12))
    return [pd.Series(i).autocorr(lag=lag) for i in arr]

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
    
    print(f"Saved data to data/{pegelname}_autokorr.csv")    
    
    plot_acf(lags, ac_raw, lower_conf=lower_conf, upper_conf=upper_conf, fn_extension="raw")
    plot_acf(lags, ac_normiert,lower_conf=lower_conf, upper_conf=upper_conf, fn_extension="normiert")

    # calculate components
    df["autokorr_saisonfigur"] = np.tile(monthly_autocorr(df=df), 40)
    df["autokorr"] = df["autokorr_saisonfigur"] * df["normiert"]
