import pandas as pd
import numpy as np
import scipy
import pymannkendall as mk

from utils.binned_stats import mean
from utils.primary_stats import hyd_years

def __preprocess(df: pd.DataFrame, which: str):
    if which == "monthly":
        x = df["Durchfluss_m3s"].tolist()
        t = np.arange(hyd_years(df)[0], hyd_years(df)[-1]+1, 1/12)
        n = len(t)
    elif which == "yearly":
        x = mean(df, which="yearly")
        t = np.arange(hyd_years(df)[0], hyd_years(df)[-1]+1, 1)
        n = len(t)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")
    
    return x, t, n

def linreg(df: pd.DataFrame, which: str):
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

def mk_test(df: pd.DataFrame, which: str):
    if which == "monthly":
        return mk.seasonal_test(df["Durchfluss_m3s"].to_numpy(), alpha=0.05, period=12)
    elif which == "yearly":
        return mk.original_test(mean(df, which="yearly"), alpha=0.05)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")

def moving_average(df: pd.DataFrame, which: str, window: int):
    """Returns the moving average of the time series."""
    x, _, _ = __preprocess(df, which)
    return np.convolve(x, np.ones(window), "valid") / window

def detrend_signal(df: pd.DataFrame) -> None:
    """Detrend the time series."""
    df["trendber"] = np.mean(df["Durchfluss_m3s"].to_numpy()) + \
        scipy.signal.detrend(df["Durchfluss_m3s"].to_numpy(), type="linear")
    df["trend"] = df["Durchfluss_m3s"].to_numpy() - df["trendber"].to_numpy()
    return df