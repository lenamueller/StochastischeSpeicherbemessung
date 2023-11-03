import pandas as pd
import numpy as np
import scipy
from collections import Counter


def missing_values(df: pd.DataFrame):
    """Returns a dictionary with the number of missing values per column."""
    return df.isnull().sum().to_dict()

def missing_dates(df: pd.DataFrame):
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