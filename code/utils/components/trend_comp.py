import numpy as np
import pandas as pd
import scipy
import pymannkendall as mk
from types import FunctionType    

from utils.statistics import hyd_years, binned_stats
from utils.plotting import plot_trend


def __preprocess(df: pd.DataFrame, which: str):
    if which == "monthly":
        x = df["Durchfluss_m3s"].tolist()
        t = np.arange(hyd_years(df)[0], hyd_years(df)[-1]+1, 1/12)
        n = len(t)
    elif which == "yearly":
        x = binned_stats(df, var="Durchfluss_m3s", bin="yearly", func=np.mean).tolist()
        t = np.arange(hyd_years(df)[0], hyd_years(df)[-1]+1, 1)
        n = len(t)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")
    
    return x, t, n

def linreg(df: pd.DataFrame, which: str) -> scipy.stats.linregress:
    """Returns slope, intercept, r, p, std_err of the linear 
    regression model using scipy.stats.linregress.
    
    Alternativ:
    x, t, n = __preprocess(df, which)
    mean_x = np.mean(x)
    mean_t = np.mean(t)
    sum_xt = sum([x_i * t_i for x_i, t_i in zip(x, t)])
    sum_t2 = sum(t_i**2 for t_i in t)
    slope = (sum_xt - n*mean_x*mean_t) / (sum_t2 - n*mean_t**2)
    intercept = mean_x - slope*mean_t
    Linreg = namedtuple("LineareRegression", ["slope", "intercept"])
    return Linreg(slope, intercept)
    """

    x, t, _ = __preprocess(df, which)
    return scipy.stats.linregress(t, x, alternative="two-sided")

def t_test_statistic(df: pd.DataFrame, which: str):
    """Returns test statistic for the t-test."""
    return linreg(df, which=which).slope / linreg(df, which=which).stderr

def mk_test(df: pd.DataFrame, which: str) -> FunctionType:
    """Trend test using the Mann-Kendall test."""
    if which == "monthly":
        return mk.seasonal_test(df["Durchfluss_m3s"].to_numpy(), alpha=0.05, period=12)
    elif which == "yearly":
        return mk.original_test(binned_stats(df, var="Durchfluss_m3s", bin="yearly", func=np.mean), alpha=0.05)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")

def moving_average(df: pd.DataFrame, which: str, window: int) -> np.ndarray[float]:
    """Returns the moving average of the time series."""
    x, _, _ = __preprocess(df, which)
    return np.convolve(x, np.ones(window), "valid") / window

def trend_comp(df: pd.DataFrame):
    print("\n--------------------------------------")
    print("\nBestimmung der Trendkomponente\n")    
    
    # Lin. Regression
    linreg_m = linreg(df, which="monthly")
    linreg_y = linreg(df, which="yearly")
    print("Lineare Regression (Jahreswerte):", linreg_y)
    print("Lineare Regression (Monatswerte):", linreg_m)
    print("Teststatistik lin. Regression (Jahreswerte):", np.round(t_test_statistic(df, which="yearly"), 3))
    print("Teststatistik lin. Regression (Monatswerte):", np.round(t_test_statistic(df, which="monthly"), 3))
    
    # Mann-Kendall test
    mk_m = mk.original_test(df["Durchfluss_m3s"], alpha=0.05)
    mk_y = mk.seasonal_test(df["Durchfluss_m3s"], alpha=0.05, period=12)
    print("\nMK-Test (Jahreswerte):", mk_y)
    print("MK-Test (Monatswerte):", mk_m)
    
    # Moving average
    ma_m = moving_average(df, which="monthly", window=12)
    ma_y = moving_average(df, which="yearly", window=5)
    
    plot_trend(df, 
               linreg_m=linreg_m, linreg_y=linreg_y, 
               mk_m=mk_m, mk_y=mk_y, 
               ma_m=ma_m, ma_y=ma_y
               )
    
    # calculate components
    df["trendber"] = np.mean(df["Durchfluss_m3s"].to_numpy()) + \
        scipy.signal.detrend(df["Durchfluss_m3s"].to_numpy(), type="linear")
    df["trend"] = df["Durchfluss_m3s"].to_numpy() - df["trendber"].to_numpy()