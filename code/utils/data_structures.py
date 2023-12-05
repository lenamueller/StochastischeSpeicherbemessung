import os
import pandas as pd
import numpy as np


def check_path(path) -> None:
    """Check if path exists, otherwise create it."""
    if not os.path.exists(path):
        os.makedirs(path)
    return None

def read_data(filepath: str) -> pd.DataFrame:
    """Read data from file."""
    data = pd.read_csv(filepath, skiprows=3, sep="\t", encoding="latin1")
    data.columns = ["Monat", "Durchfluss_m3s"]
    data["Monat"] = data["Monat"].astype(str)
    data["Durchfluss_m3s"] = data["Durchfluss_m3s"].astype(float)
    data["Datum"] = pd.to_datetime(data["Monat"], format="%m/%Y")
    return data

def read_gen_data() -> pd.DataFrame:
    """Read generated data from file."""
    data = pd.read_csv("data/Klingenthal_thomasfiering_timeseries_100.csv",
                       index_col=0)
    return data

def _monthly_vals(df: pd.DataFrame, i: int)-> np.ndarray:
    df = df[df["Monat"].str.startswith(str(i).zfill(2))]
    return df["Durchfluss_m3s"].to_numpy()