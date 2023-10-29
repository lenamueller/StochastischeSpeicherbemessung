import os
import numpy as np
import scipy
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt
from setup import image_path,tu_mediumblue, tu_red, tu_grey


class TimeSeries:

    def __init__(self, name: str):
        """Constructor."""
        self.name: str = name
        self.data: pd.DataFrame = None
        self._read_data(filename=f"Daten_{name}.txt")
        
    # ---------------------------------------------------------------
    # Read data
    # ---------------------------------------------------------------
    
    def _read_data(self, filename: str):
        """Read data from file."""
        filepath = "data/" + filename
        self.data = pd.read_csv(filepath, skiprows=3, sep="\t", encoding="latin1")
        self.data.columns = ["Monat", "Durchfluss_m3s"]
        self.data["Monat"] = self.data["Monat"].astype(str)
        self.data["Durchfluss_m3s"] = self.data["Durchfluss_m3s"].astype(float)
        self.data["Datum"] = pd.to_datetime(self.data["Monat"], format="%m/%Y")
    
    # ---------------------------------------------------------------
    # Primäre Informationen
    # ---------------------------------------------------------------
    
    def sample_number(self):
        """Returns the sample number."""
        return len(self.data)
    
    def earliest_date(self):
        """Returns the earliest date."""
        return self.data["Datum"].min()
    
    def latest_date(self):
        """Returns the latest date."""
        return self.data["Datum"].max()
    
    # ---------------------------------------------------------------
    # Primärstatistik
    # ---------------------------------------------------------------
    
    def min_val(self):
        """Returns the minimum value."""
        return self.data["Durchfluss_m3s"].min()
    
    def min_val_month(self):
        """Returns the month of the minimum value."""
        min_index = self.data["Durchfluss_m3s"].idxmin()
        return self.data["Monat"].iloc[min_index]
    
    def max_val_month(self):
        """Returns the month of the maximum value."""
        max_index = self.data["Durchfluss_m3s"].idxmax()
        return self.data["Monat"].iloc[max_index]
    
    def max_val(self):
        """Returns the maximum value."""
        return self.data["Durchfluss_m3s"].max()
    
    def first_central_moment(self):
        """Returns the first central moment."""
        return self.data["Durchfluss_m3s"].mean()
    
    def second_central_moment(self):
        """Returns the second central moment."""
        return self.data["Durchfluss_m3s"].var()
    
    def third_central_moment(self):
        """Returns the third central moment."""
        return scipy.stats.skew(self.data["Durchfluss_m3s"], bias=True)
    
    def fourth_central_moment(self):
        """Returns the fourth central moment."""
        return scipy.stats.kurtosis(self.data["Durchfluss_m3s"], bias=True)
    
    def standard_deviation_biased(self):
        """Returns the biased standard deviation."""
        return np.sqrt(self.second_central_moment())
    
    def standard_deviation_unbiased(self):
        """Returns the unbiased standard deviation."""
        return np.sqrt(self.second_central_moment() * \
            (self.sample_number() / (self.sample_number() - 1)))
    
    def skewness_biased(self):
        """Returns the biased skewness."""
        mean = self.first_central_moment()
        std = self.standard_deviation_biased()
        n = self.sample_number()
        return np.sum(((self.data["Durchfluss_m3s"] - mean)/std)**3) / n
    
    def skewness_unbiased(self):
        """Returns the unbiased skewness."""
        n = self.sample_number()
        return self.skewness_biased() * n/(n-1) * (n-1)/(n-2)

    def kurtosis_biased(self):
        """Returns the biased kurtosis."""
        mean = self.first_central_moment()
        std = self.standard_deviation_biased()
        n = self.sample_number()
        return np.sum(((self.data["Durchfluss_m3s"] - mean)/std)**4) / n - 3
    
    def kurtois_unbiased(self):
        """Returns the unbiased kurtosis."""
        n = self.sample_number()
        return self.kurtosis_biased() * n/(n-1) * (n-1)/(n-2) * (n-2)/(n-3)        

    def quartile(self, q: int):
        """Returns the q-th quartile."""
        return self.data["Durchfluss_m3s"].quantile(q=q)

    def iqr(self):
        """"Returns the interquartile range."""
        return self.data["Durchfluss_m3s"].quantile(q=0.75) - \
                self.data["Durchfluss_m3s"].quantile(q=0.25)

    # ---------------------------------------------------------------
    # Hydrologische Kennwerte
    # ---------------------------------------------------------------
    
    def hydro_values(self):
        """Returns a dictionary with the hydrological values."""
        
        hydro_parameters = {
            "HHQ": (None, None),
            "MHQ": None,
            "MQ": None,
            "MNQ": None,
            "NNQ": (None, None),
        }
        
        # NNQ
        min_value = min(self.data["Durchfluss_m3s"])
        min_index = self.data["Durchfluss_m3s"].idxmin()
        min_Monat = self.data["Monat"].iloc[min_index]
        hydro_parameters["NNQ"] = (min_value, min_Monat)
        
        # HHQ
        max_value = max(self.data["Durchfluss_m3s"])
        max_index = self.data["Durchfluss_m3s"].idxmax()
        max_Monat = self.data["Monat"].iloc[max_index]
        hydro_parameters["HHQ"] = (max_value, max_Monat)
        
        # MHQ, MNQ, MQ
        hydrological_years = sorted(self.data["Datum"].dt.year.unique())[1:]
        highest_q = []
        lowest_q = []
        mean_q = []
        for i in range(len(hydrological_years)):
            year = hydrological_years[i]
            start_date = pd.Timestamp(year-1, 10, 1)
            end_date = pd.Timestamp(year, 9, 30)
            subset = self.data.loc[(self.data["Datum"] >= start_date) & \
                    (self.data["Datum"] <= end_date)]
            highest_q.append(subset["Durchfluss_m3s"].max())
            lowest_q.append(subset["Durchfluss_m3s"].min())
            mean_q.append(subset["Durchfluss_m3s"].mean())

        hydro_parameters["MHQ"] = np.mean(highest_q)
        hydro_parameters["MNQ"] = np.mean(lowest_q)
        hydro_parameters["MQ"] = np.mean(mean_q)

        return hydro_parameters

    # ---------------------------------------------------------------
    # Konsistenz - Check    
    # ---------------------------------------------------------------
    
    def missing_values(self):
        """Returns a dictionary with the number of missing values per column."""
        return self.data.isnull().sum().to_dict()

    def missing_dates(self):
        """Returns a list of missing dates."""
        min_date = self.data["Datum"].min()
        max_date = self.data["Datum"].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq="MS")
        missing_dates = date_range.difference(self.data["Datum"])
        return [str(date) for date in missing_dates]

    def duplicates(self) -> list[pd.Timestamp]:
        """Find indexes of duplicates."""
        if self.data.empty:
            raise ValueError("empty data")
        days = self.data.index.tolist()
        return [item for item, count in Counter(days).items() if count > 1]

    # ---------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------
    
    def plot_raw(self):
        """Plot raw data."""
        
        if not os.path.exists(image_path):
            os.makedirs(image_path)
            
        max_value = self.max_val()
        max_month = self.max_val_month()
        min_value = self.min_val()
        min_month = self.min_val_month()

        plt.figure(figsize=(10, 5))
        plt.plot(self.data["Monat"], self.data["Durchfluss_m3s"], 
                 c=tu_mediumblue, linewidth=0.8, label="Rohdaten")
        plt.axhline(y=max_value, c=tu_red, linestyle="--", linewidth=0.8, 
                    label=f"Max: {max_month}: {max_value} m³/s")
        plt.axhline(y=min_value, c=tu_grey, linestyle="--", linewidth=0.8, 
                    label=f"Min: {min_month}: {min_value} m³/s")
        plt.scatter(max_month, max_value, marker="o", 
                    facecolors='none', edgecolors=tu_red, s=30)
        plt.scatter(min_month, min_value, marker="o", 
                    facecolors='none', edgecolors=tu_grey, s=30)
        plt.xlabel("Monat")
        plt.ylabel("Durchfluss [m³/s]")
        plt.xticks(self.data["Monat"][::12], rotation=90)
        plt.yticks(np.arange(0, max_value, 1), minor=False)
        plt.yticks(np.arange(0, max_value, 0.25), minor=True)
        plt.grid(which="major", axis="x", color="grey", alpha=0.15)
        plt.grid(which="major", axis="y", color="grey", alpha=0.75)
        plt.grid(which="minor", axis="y", color="grey", alpha=0.15)
        plt.ylim(bottom=0)
        plt.xlim(left=self.data["Monat"].min(), right=self.data["Monat"].max())
        plt.legend(loc="upper right")
        
        plt.savefig(image_path+f"{self.name}_raw.png", dpi=300, bbox_inches="tight")
        
        return None
    
    def plot_hist(self):
        """Plot histogram of raw data."""
        
        if not os.path.exists(image_path):
            os.makedirs(image_path)
            
        max_value = self.max_val()
        max_index = self.data["Durchfluss_m3s"].idxmax()
        max_month = self.data["Monat"].iloc[max_index]
        min_value = min_value = self.min_val()
        min_index = self.data["Durchfluss_m3s"].idxmin()
        min_month = self.data["Monat"].iloc[min_index]

        plt.figure(figsize=(10, 5))
        plt.hist(self.data["Durchfluss_m3s"], bins=np.arange(0, max_value+0.1, 0.1), 
                density=False, 
                color=tu_mediumblue, label="Empirische Verteilung",
                lw=0.8, edgecolor="black", alpha=0.8)
        plt.xticks(np.arange(0, max_value+1, 1), minor=False)
        plt.xticks(np.arange(0, max_value+0.1, 0.1), minor=True)
        plt.xlim(left=0, right=round(max_value, 1)+0.1)
        plt.xlabel("Durchfluss [m³/s]")
        plt.ylabel("empirische Häufigkeit")
        plt.grid(which="minor", axis="y", color="grey", alpha=0.15)
        plt.grid(which="major", axis="y", color="grey", alpha=0.75)
        plt.grid(which="minor", axis="x", color="grey", alpha=0.15)
        plt.grid(which="major", axis="x", color="grey", alpha=0.75)
        plt.xlim(left=0)
        plt.twinx()
        plt.ylabel("Dichte")
        plt.savefig(image_path+f"{self.name}_hist.png", dpi=300, bbox_inches="tight")
        
        return None
    
    