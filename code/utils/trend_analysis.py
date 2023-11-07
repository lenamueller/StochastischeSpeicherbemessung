import pandas as pd
import numpy as np
import scipy
from collections import namedtuple

from utils.binned_stats import mean
from utils.primary_stats import hyd_years


def linreg(df: pd.DataFrame, which: str):
    """Returns slope, intercept, r, p, std_err of the linear regression model."""
    if which == "monthly":
        x = mean(df, which="monthly")
        t = np.arange(1, len(x)+1, 1)
    elif which == "yearly":
        x = mean(df, which="yearly")
        t = hyd_years(df)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")
    return scipy.stats.linregress(t, x, alternative="two-sided")

def coeff_of_determination(df: pd.DataFrame, which: str):
    """Returns the coefficient of determination."""
    if which == "monthly":
        _, _, r, _, _ = linreg(df, which="monthly")
        return r**2
    elif which == "yearly":
        _, _, r, _, _ = linreg(df, which="yearly")
        return r**2
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")

def linreg_ab(df: pd.DataFrame, which: str):
    """Returns slope and intercept of the linear regression model."""
    
    if which == "monthly":
        x = mean(df, which="monthly")
        t = np.arange(1, len(x)+1, 1)
    elif which == "yearly":
        x = mean(df, which="yearly")
        t = hyd_years(df)
    
    n = len(df)    
    mean_x = np.mean(x)
    mean_t = np.mean(t)
    sum_xt = sum([x_i * t_i for x_i, t_i in zip(x, t)])
    sum_t2 = sum(t_i**2 for t_i in t)
    slope = (sum_xt - n*mean_x*mean_t) / (sum_t2 - n*mean_t**2)
    intercept = mean_x - slope*mean_t
    Linreg = namedtuple("LineareRegression", ["slope", "intercept"])
    l = Linreg(slope, intercept)
    return l

def trendtest(df: pd.DataFrame):
    # TODO: #2 Implement t-test for trend
    pass

def detrend_signal(df: pd.DataFrame) -> None:
    """Detrend the time series."""
    # TODO: #3 Check algorithm for detrending
    data = df["Durchfluss_m3s"].to_numpy()
    return np.mean(data) + scipy.signal.detrend(data, type="linear")
