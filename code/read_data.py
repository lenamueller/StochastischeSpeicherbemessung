import pandas as pd


def read_data(filename) -> pd.DataFrame:
    """Reads data from a file and returns a pandas DataFrame."""
    filepath = "data/" + filename
    data = pd.read_csv(filepath, skiprows=3, sep="\t", encoding="latin1")
    data.columns = ["Monat", "Durchfluss"]
    data["Monat"] = pd.to_datetime(data["Monat"], format="%m/%Y").dt.strftime("%m/%Y")
    return data