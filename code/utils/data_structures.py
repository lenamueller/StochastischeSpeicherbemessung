import os
import numpy as np
import pandas as pd


def check_path(path):
    """Check if path exists, otherwise create it."""
    if not os.path.exists(path):
        os.makedirs(path)
    return None

def read_data(filepath: str):
    """Read data from file."""
    data = pd.read_csv(filepath, skiprows=3, sep="\t", encoding="latin1")
    data.columns = ["Monat", "Durchfluss_m3s"]
    data["Monat"] = data["Monat"].astype(str)
    data["Durchfluss_m3s"] = data["Durchfluss_m3s"].astype(float)
    data["Datum"] = pd.to_datetime(data["Monat"], format="%m/%Y")
    return data
