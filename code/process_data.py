import pandas as pd
import numpy as np


def accumulate_yearly(data: pd.DataFrame):
    data_acc = pd.DataFrame()
    years = data["Monat"].str.split("/").str[1].unique()
    hyd_years = np.arange(int(years[0])+1, int(years[-1])+1)
    yearly_vals = np.zeros(len(hyd_years))
    for i in range(len(hyd_years)):
        yearly_vals[i] = data["Durchfluss"].iloc[12*i:12*i+12].mean()    
    data_acc["Jahr"] = hyd_years
    data_acc["Durchfluss"] = yearly_vals
    return data_acc