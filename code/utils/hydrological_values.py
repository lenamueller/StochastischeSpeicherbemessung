import pandas as pd
import numpy as np


def hydro_values(df: pd.DataFrame):
    """Returns a dictionary with the hydrological values."""
    
    hydro_parameters = {
        "HHQ": (None, None),
        "MHQ": None,
        "MQ": None,
        "MNQ": None,
        "NNQ": (None, None),
    }
    
    # NNQ
    min_value = min(df["Durchfluss_m3s"])
    min_index = df["Durchfluss_m3s"].idxmin()
    min_Monat = df["Monat"].iloc[min_index]
    hydro_parameters["NNQ"] = (min_value, min_Monat)
    
    # HHQ
    max_value = max(df["Durchfluss_m3s"])
    max_index = df["Durchfluss_m3s"].idxmax()
    max_Monat = df["Monat"].iloc[max_index]
    hydro_parameters["HHQ"] = (max_value, max_Monat)
    
    # MHQ, MNQ, MQ
    hydrological_years = sorted(df["Datum"].dt.year.unique())[1:]
    highest_q = []
    lowest_q = []
    mean_q = []
    for i in range(len(hydrological_years)):
        year = hydrological_years[i]
        start_date = pd.Timestamp(year-1, 10, 1)
        end_date = pd.Timestamp(year, 9, 30)
        subset = df.loc[(df["Datum"] >= start_date) & \
                (df["Datum"] <= end_date)]
        highest_q.append(subset["Durchfluss_m3s"].max())
        lowest_q.append(subset["Durchfluss_m3s"].min())
        mean_q.append(subset["Durchfluss_m3s"].mean())

    hydro_parameters["MHQ"] = np.mean(highest_q)
    hydro_parameters["MNQ"] = np.mean(lowest_q)
    hydro_parameters["MQ"] = np.mean(mean_q)

    return hydro_parameters