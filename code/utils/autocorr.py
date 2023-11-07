import pandas as pd
import numpy as np

from config import pegelname


def autocorr(df: pd.DataFrame):
    """Returns the autocorrelation function."""
    lags = np.arange(51)
    corr = [df["saisonber"].autocorr(lag=i) for i in lags]
    lower_conf = -1.96 / np.sqrt(len(df))
    upper_conf = 1.96 / np.sqrt(len(df))
    return lags, corr, lower_conf, upper_conf
