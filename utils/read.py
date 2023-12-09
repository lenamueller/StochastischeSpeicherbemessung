import pandas as pd


def read_raw_data(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath, skiprows=3, sep="\t", encoding="latin1")
    data.columns = ["Monat", "Durchfluss_m3s"]
    data["Monat"] = data["Monat"].astype(str)
    data["Durchfluss_m3s"] = data["Durchfluss_m3s"].astype(float)
    return data

def monthly_vals(df: pd.DataFrame, month: int) -> list:
    df = df[df["Monat"].str.startswith(str(month).zfill(2))]
    return df["Durchfluss_m3s"].to_list()