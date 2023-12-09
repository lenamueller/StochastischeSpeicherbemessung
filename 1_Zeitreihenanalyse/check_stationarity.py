import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> adfuller:
    """Test for stationarity using the Augmented Dickey-Fuller test."""
    return adfuller(df[var])