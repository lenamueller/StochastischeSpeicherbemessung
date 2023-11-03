import pandas as pd
import numpy as np
import scipy

from binned_stats import mean
from primary_stats import hyd_years


def linreg_monthly(df: pd.DataFrame):
    # todo
    t = np.arange(1, len(x)+1, 1)
    x = df["Durchfluss_m3s"].to_numpy()
    return scipy.stats.linregress(t, x)

def linreg_yearly(df: pd.DataFrame):
    # todo
    t = hyd_years(df)
    x = mean(df, which="yearly")
    return scipy.stats.linregress(t, x)

def trendtest(df: pd.DataFrame):
    # todo
    pass

def detrend_signal(df: pd.DataFrame) -> None:
    """Detrend the time series."""
    # todo check algorithm
    data = df["Durchfluss_m3s"].to_numpy()
    return np.mean(data) + scipy.signal.detrend(data, type="linear")