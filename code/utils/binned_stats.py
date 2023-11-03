import numpy as np
import scipy
import pandas as pd

from data_structures import df_to_np


def mean(df: pd.DataFrame, which: str):
    """Returns a list of montly or yearly means."""
    data_np = df_to_np(df)
    if which == "monthly":
        return np.mean(data_np, axis=0)
    elif which == "yearly":
        return np.mean(data_np, axis=1)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")

def median(df: pd.DataFrame, which: str):
    """Returns a list of montly or yearly medians."""
    data_np = df_to_np(df)
    if which == "monthly":
        return np.median(data_np, axis=0)
    elif which == "yearly":
        return np.median(data_np, axis=1)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")
    
def variance(df: pd.DataFrame, which: str):
    """Returns a list of montly or yearly variances."""
    data_np = df_to_np(df)
    if which == "monthly":
        return np.var(data_np, axis=0)
    elif which == "yearly":
        return np.var(data_np, axis=1)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")

def skewness(df: pd.DataFrame, which: str):
    """Returns a list of montly or yearly skewness."""
    data_np = df_to_np(df)
    if which == "monthly":
        return scipy.stats.skew(data_np, axis=0, bias=True)
    elif which == "yearly":
        return scipy.stats.skew(data_np, axis=1, bias=True)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")