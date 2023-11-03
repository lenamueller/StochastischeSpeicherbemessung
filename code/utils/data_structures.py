import numpy as np
import pandas as pd


def read_data(filename: str):
    """Read data from file."""
    filepath = "data/" + filename
    data = pd.read_csv(filepath, skiprows=3, sep="\t", encoding="latin1")
    data.columns = ["Monat", "Durchfluss_m3s"]
    data["Monat"] = data["Monat"].astype(str)
    data["Durchfluss_m3s"] = data["Durchfluss_m3s"].astype(float)
    data["Datum"] = pd.to_datetime(data["Monat"], format="%m/%Y")
    return data

def df_to_np(df: pd.DataFrame):
    data_np = df["Durchfluss_m3s"].to_numpy()
    return  np.reshape(data_np, (-1, 12))