import scipy
import numpy as np
import pandas as pd


def sample_number(df: pd.DataFrame):
    """Returns the sample number."""
    return len(df)

def earliest_date(df: pd.DataFrame):
    """Returns the earliest date."""
    return df["Datum"].min()

def latest_date(df: pd.DataFrame):
    """Returns the latest date."""
    return df["Datum"].max()

def years(df: pd.DataFrame):
    """Returns a list of the years."""
    return df["Datum"].dt.year.unique()

def hyd_years(df: pd.DataFrame):
    years = df["Datum"].dt.year.unique()
    return years[1:]

def min_val(df: pd.DataFrame):
    """Returns the minimum value."""
    return df["Durchfluss_m3s"].min()

def min_val_month(df: pd.DataFrame):
    """Returns the month of the minimum value."""
    min_index = df["Durchfluss_m3s"].idxmin()
    return df["Monat"].iloc[min_index]

def max_val_month(df: pd.DataFrame):
    """Returns the month of the maximum value."""
    max_index = df["Durchfluss_m3s"].idxmax()
    return df["Monat"].iloc[max_index]

def max_val(df: pd.DataFrame):
    """Returns the maximum value."""
    return df["Durchfluss_m3s"].max()

def first_central_moment(df: pd.DataFrame):
    """Returns the first central moment."""
    return df["Durchfluss_m3s"].mean()

def second_central_moment(df: pd.DataFrame):
    """Returns the second central moment."""
    return df["Durchfluss_m3s"].var()

def third_central_moment(df: pd.DataFrame):
    """Returns the third central moment."""
    return scipy.stats.skew(df["Durchfluss_m3s"], bias=True)

def fourth_central_moment(df: pd.DataFrame):
    """Returns the fourth central moment."""
    return scipy.stats.kurtosis(df["Durchfluss_m3s"], bias=True)

def standard_deviation_biased(df: pd.DataFrame):
    """Returns the biased standard deviation."""
    return np.sqrt(second_central_moment(df))

def standard_deviation_unbiased(df: pd.DataFrame):
    """Returns the unbiased standard deviation."""
    return np.sqrt(second_central_moment(df) * \
        (sample_number(df) / (sample_number(df) - 1)))

def skewness_biased(df: pd.DataFrame):
    """Returns the biased skewness."""
    mean = first_central_moment(df)
    std = standard_deviation_biased(df)
    n = sample_number(df)
    return np.sum(((df["Durchfluss_m3s"] - mean)/std)**3) / n

def skewness_unbiased(df: pd.DataFrame):
    """Returns the unbiased skewness."""
    n = sample_number(df)
    return skewness_biased(df) * n/(n-1) * (n-1)/(n-2)

def kurtosis_biased(df: pd.DataFrame):
    """Returns the biased kurtosis."""
    mean = first_central_moment(df)
    std = standard_deviation_biased(df)
    n = sample_number(df)
    return np.sum(((df["Durchfluss_m3s"] - mean)/std)**4) / n - 3

def kurtosis_unbiased(df: pd.DataFrame):
    """Returns the unbiased kurtosis."""
    n = sample_number(df)
    return kurtosis_biased(df) * n/(n-1) * (n-1)/(n-2) * (n-2)/(n-3)        

def quartile(df: pd.DataFrame, q: int):
    """Returns the q-th quartile."""
    return df["Durchfluss_m3s"].quantile(q=q)

def iqr(df: pd.DataFrame):
    """"Returns the interquartile range."""
    return df["Durchfluss_m3s"].quantile(q=0.75) - \
            df["Durchfluss_m3s"].quantile(q=0.25)