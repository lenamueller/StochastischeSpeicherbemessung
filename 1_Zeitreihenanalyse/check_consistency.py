import numpy as np
import pandas as pd
from collections import Counter, namedtuple


def missing_values(df: pd.DataFrame) -> dict:
    """Returns a dictionary with the number of missing values per column."""
    return df.isnull().sum().to_dict()

def missing_dates(df: pd.DataFrame) -> list[str]:
    """Returns a list of missing dates."""
    min_date = df["Datum"].min()
    max_date = df["Datum"].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq="MS")
    missing_dates = date_range.difference(df["Datum"])
    return [str(date) for date in missing_dates]

def duplicates(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Find indexes of duplicates."""
    if df.empty:
        raise ValueError("empty data")
    days = df.index.tolist()
    return [item for item, count in Counter(days).items() if count > 1]

def outlier_test_iqr(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> tuple[float, float, pd.DataFrame]:
    """Returns a list of outliers using the IQR method."""
    q1 = df[var].quantile(q=0.25, interpolation="nearest")
    q3 = df[var].quantile(q=0.75, interpolation="nearest")
    iqr = q3 - q1
    g_upper = q3 + 1.5*iqr
    g_lower = q1 - 1.5*iqr
    OutlierTestIQR = namedtuple("OutlierTestIQR", 
                     ["upper_bound", "lower_bound", "outlier"])
    return OutlierTestIQR(g_upper, g_lower, df.loc[(df[var] < g_lower) | (df[var] > g_upper)])
    
def outlier_test_zscore(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> tuple[float, float, pd.DataFrame]:
    """Returns a list of outliers using the z-score method."""
    g_upper = df[var].mean() + 3*df[var].std()
    g_lower = df[var].mean() - 3*df[var].std()
    OutlierTestZscore = namedtuple("OutlierTestZscore",
                                   ["upper_bound", "lower_bound", "outlier"])
    return OutlierTestZscore(g_upper, g_lower, df.loc[(df[var] < g_lower) | (df[var] > g_upper)])

def outlier_test_grubbs(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> tuple[float, pd.DataFrame]:
    """Returns a list of outliers using the Grubbs method."""
    max_diff = np.max(np.abs(df[var] - df[var].mean()))
    s = np.std(df[var])
    g = max_diff / s
    OutlierTestGrubbs = namedtuple("OutlierTestGrubbs",
                                   ["bound", "outlier"])
    return OutlierTestGrubbs(g, df.loc[df[var] > g])
