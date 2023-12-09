import numpy as np
import pandas as pd


def confidence_interval(df: pd.DataFrame, lags: list[float]) -> tuple[list[float], list[float]]:
    """Returns the confidence interval."""
    k = lags
    n = len(df)
    T_ALPHA = 1.645 # alpha = 0.05
    lower_conf = (-1 - T_ALPHA*np.sqrt(n-k-1)) / (n-k+1)
    upper_conf = (1 + T_ALPHA*np.sqrt(n-k-1)) / (n-k+1)
    return lower_conf, upper_conf
