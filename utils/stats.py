import scipy
import numpy as np
import pandas as pd
from types import FunctionType    


# -----------------------------------------
#           Primary statistics
# -----------------------------------------

def sample_number(df: pd.DataFrame) -> int:
    """Returns the sample number."""
    return len(df)

def earliest_date(df: pd.DataFrame) -> str:
    """Returns the earliest date."""
    return df["Datum"].min()

def latest_date(df: pd.DataFrame) -> str:
    """Returns the latest date."""
    return df["Datum"].max()

def years(df: pd.DataFrame) -> list[int]:
    """Returns a list of the years."""
    return df["Datum"].dt.year.unique()

def hyd_years(df: pd.DataFrame) -> list[int]:
    years = df["Datum"].dt.year.unique()
    return years[1:]

def min_val(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> tuple[float, str]:
    """Returns the minimum value and the month of it."""
    val = df[var].min()
    min_index = df[var].idxmin()
    month =  df["Monat"].iloc[min_index]
    return val, month

def max_val(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> tuple[float, str]:
    """Returns the maximum value and the month of it."""
    val = df[var].max()
    max_index = df[var].idxmax()    
    month = df["Monat"].iloc[max_index]
    return val, month

def central_moment(df: pd.DataFrame, nth: int, var: str = "Durchfluss_m3s") -> float | ValueError:
    """Returns the n-th central moment. nth must be 1, 2, 3 or 4."""
    mean = df[var].mean()
    diff = [(i-mean)**nth for i in df[var]]
    
    if nth == 1:
        return mean
    elif nth == 2:
        return np.sum(diff) / sample_number(df)
    elif nth == 3:
        return np.sum(diff) / sample_number(df)
    elif nth == 4:
        return np.sum(diff) / sample_number(df)
    else:
        raise ValueError("n must be 1, 2, 3 or 4")

def standard_deviation(df: pd.DataFrame, bias: bool, var: str = "Durchfluss_m3s") -> float:
    """Returns the standard deviation."""
    if bias:
        return np.sqrt(central_moment(df, nth=2, var=var))
    else:
        return np.sqrt(central_moment(df, nth=2, var=var) * \
        (sample_number(df) / (sample_number(df) - 1)))

def skewness(df: pd.DataFrame, bias: bool, var: str = "Durchfluss_m3s") -> float:
    """Returns the biased skewness."""
    std = standard_deviation(df, bias=True)
    n = sample_number(df)
    biased = np.sum(((df[var] - central_moment(df, nth=1, var=var))/std)**3) / n
    if bias:
        return biased
    else: 
        return biased * n/(n-1) * (n-1)/(n-2)

def kurtosis(df: pd.DataFrame, bias: bool, var: str = "Durchfluss_m3s") -> float:
    """Returns the biased kurtosis."""
    std = standard_deviation(df, bias=True)
    n = sample_number(df)
    biased = np.sum(((df[var] - central_moment(df, nth=1, var=var))/std)**4) - 3
    if bias: 
        return biased
    else:
        return biased * n/(n-1) * (n-1)/(n-2) * (n-2)/(n-3)        

def quantiles(df: pd.DataFrame, q: float, var: str = "Durchfluss_m3s") -> float:
    return df[var].quantile(q=q, interpolation="nearest")
    
def iqr(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> float:
    """"Returns the interquartile range using nearest rank method."""
    return df[var].quantile(q=0.75, interpolation="nearest") - \
            df[var].quantile(q=0.25, interpolation="nearest")

def autocorrelation(df: pd.DataFrame, var: str, lag: int = 1) -> float:
    """Returns the autocorrelation function."""
    return pd.Series(df[var]).autocorr(lag=lag)

# -----------------------------------------
#    binned statistics (monthly, yearly)
# -----------------------------------------

def binned_stats(
        df: pd.DataFrame, 
        var: str, 
        bin: str, 
        func: FunctionType
        ) -> list[float]:
    """
    Returns monthly or yearly statistical values.
    
    Possible functions are:
    - np.mean
    - np.var
    - np.std
    - scipy.stats.skew
    - np.sum
    """
    arr = np.reshape(df[var].to_numpy(), (-1, 12))
    d = {"monthly": 0, "yearly": 1}
    return func(arr, axis=d[bin])


def monthly_autocorr(df: pd.DataFrame, var: str = "saisonber", which: str = "maniak") -> list[float]:
    """Returns a list of monthly autocorrelations for lag (k) = 1."""

    months = df["Monat"]
    pairs = [("11", "12"), ("12", "01"), ("01", "02"), ("02", "03"),
             ("03", "04"), ("04", "05"), ("05", "06"), ("06", "07"),
             ("07", "08"), ("08", "09"), ("09", "10"), ("10", "11")]
    
    coeff = []
    for i in range(12):
        first_month, second_month = pairs[i]
        x_i = df[var][months.str.startswith(first_month)].tolist()
        x_ik = df[var][months.str.startswith(second_month)].tolist()
        assert len(x_i) == len(x_ik)
        
        mean_x_i = np.mean(x_i)
        mean_x_ik = np.mean(x_ik)
        std_x_i = np.std(x_i)
        std_x_ik = np.std(x_ik)
        k = 1
        n = len(x_i)
        
        if which == "pearson":
            coeff.append(scipy.stats.stats.pearsonr(x_i, x_ik).statistic)
        elif which == "maniak":
            prod = [(i - mean_x_i) * (ik - mean_x_ik) for i, ik in zip(x_i, x_ik)]
            r_k_maniak = (sum(prod[:-k])) / (std_x_i*std_x_ik) / (n-k)
            coeff.append(r_k_maniak)
        else:
            raise ValueError("which must be 'pearson' or 'maniak'")
    return coeff

def yearly_autocorr(
    df: pd.DataFrame,
    lag: int,
    var: str = "Durchfluss_m3s") -> list[float]:
    """Returns a list of yearly autocorrelations."""
    arr = np.reshape(df[var].to_numpy(), (-1, 12))
    return [pd.Series(i).autocorr(lag=lag) for i in arr]

# -----------------------------------------
#           Hydrological values
# -----------------------------------------

def hydro_values(df: pd.DataFrame) -> dict[str, tuple[float, str] | float]:
    """Returns a dictionary with the hydrological values."""
    
    hydro_parameters = {
        "HHQ": (None, None),
        "MHQ": None,
        "MQ": None,
        "MNQ": None,
        "NNQ": (None, None),
    }
    
    # NNQ
    min_value, min_Monat = min_val(df, var="Durchfluss_m3s")
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