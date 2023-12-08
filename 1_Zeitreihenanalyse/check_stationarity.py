import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> adfuller:
    """Test for stationarity using the Augmented Dickey-Fuller test."""
    return adfuller(df[var])

def stationarity_check(df: pd.DataFrame, var: str = "Durchfluss"):
    print("\n--------------------------------------")
    print("\n\tStationaritätsprüfung\n")
    print("ADF-Test:", adf_test(df))