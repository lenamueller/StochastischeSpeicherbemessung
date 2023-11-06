import pandas as pd
import numpy as np
import scipy

from utils.binned_stats import mean
from utils.primary_stats import hyd_years


def linreg_monthly(df: pd.DataFrame):
    # TODO: #1 Check algorithm (also for linreg_yearly)
    x = df["Durchfluss_m3s"].to_numpy()
    t = np.arange(1, len(x)+1, 1)
    return scipy.stats.linregress(t, x)

def linreg_yearly(df: pd.DataFrame):
    # TODO #1: Check algorithm
    x = mean(df, which="yearly")
    t = hyd_years(df)
    return scipy.stats.linregress(t, x)

def trendtest(df: pd.DataFrame):
    # TODO: #2 Implement t-test for trend
    pass

def detrend_signal(df: pd.DataFrame) -> None:
    """Detrend the time series."""
    # TODO: #3 Check algorithm for detrending
    data = df["Durchfluss_m3s"].to_numpy()
    return np.mean(data) + scipy.signal.detrend(data, type="linear")