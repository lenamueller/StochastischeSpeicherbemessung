import scipy
import numpy as np
import pandas as pd
import pymannkendall as mk


# -----------------------------------------
#           Primary statistics
# -----------------------------------------

def sample_number(df: pd.DataFrame):
    """Returns the sample number."""
    return len(df)

def earliest_date(df: pd.DataFrame):
    """Returns the earliest date."""
    return df["Datum"].min()

def latest_date(df: pd.DataFrame):
    """Returns the latest date."""
    return df["Datum"].max()

def years(df: pd.DataFrame):
    """Returns a list of the years."""
    return df["Datum"].dt.year.unique()

def hyd_years(df: pd.DataFrame):
    years = df["Datum"].dt.year.unique()
    return years[1:]

def min_val(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the minimum value."""
    return df[var].min()

def min_val_month(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the month of the minimum value."""
    min_index = df[var].idxmin()
    return df["Monat"].iloc[min_index]

def max_val_month(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the month of the maximum value."""
    max_index = df[var].idxmax()
    return df["Monat"].iloc[max_index]

def max_val(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the maximum value."""
    return df[var].max()

def first_central_moment(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the first central moment (mean)."""
    return df[var].mean()

def second_central_moment(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the second central moment (variance)."""
    n = sample_number(df)
    mean = first_central_moment(df)
    diff = [(i-mean)**2 for i in df[var]]
    return np.sum(diff) / n

def third_central_moment(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the third central moment."""
    n = sample_number(df)
    mean = first_central_moment(df)
    diff = [(i-mean)**3 for i in df[var]]
    return np.sum(diff) / n

def fourth_central_moment(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the fourth central moment."""
    n = sample_number(df)
    mean = first_central_moment(df)
    diff = [(i-mean)**4 for i in df[var]]
    return np.sum(diff) / n

def standard_deviation_biased(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the biased standard deviation."""
    return np.sqrt(second_central_moment(df, var))

def standard_deviation_unbiased(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the unbiased standard deviation."""
    return np.sqrt(second_central_moment(df, var) * \
        (sample_number(df) / (sample_number(df) - 1)))

def skewness_biased(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the biased skewness."""
    mean = first_central_moment(df)
    std = standard_deviation_biased(df)
    n = sample_number(df)
    return np.sum(((df[var] - mean)/std)**3) / n

def skewness_unbiased(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the unbiased skewness."""
    n = sample_number(df)
    return skewness_biased(df, var) * n/(n-1) * (n-1)/(n-2)

def kurtosis_biased(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the biased kurtosis."""
    mean = first_central_moment(df)
    std = standard_deviation_biased(df)
    res = np.sum(((df[var] - mean)/std)**4) - 3
    return res

def kurtosis_unbiased(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the unbiased kurtosis."""
    n = sample_number(df)
    return kurtosis_biased(df, var) * n/(n-1) * (n-1)/(n-2) * (n-2)/(n-3)        

def quartile(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns the first, second or third quartile."""
    return (
        df[var].quantile(q=0.25, interpolation="nearest"),
        df[var].quantile(q=0.5, interpolation="nearest"),
        df[var].quantile(q=0.75, interpolation="nearest")
    )
    
def iqr(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """"Returns the interquartile range using nearest rank method."""
    return df[var].quantile(q=0.75, interpolation="nearest") - \
            df[var].quantile(q=0.25, interpolation="nearest")

# -----------------------------------------
#           Hydrological values
# -----------------------------------------

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

# -----------------------------------------
#    binned statistics (monthly, yearly)
# -----------------------------------------

def monthly_mean(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns a list of monthly means."""
    arr = np.reshape(df[var].to_numpy(), (-1, 12))
    return np.mean(arr, axis=0)

def yearly_mean(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns a list of yearly means."""
    arr = np.reshape(df[var].to_numpy(), (-1, 12))
    return np.mean(arr, axis=1)

def monthly_variance(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns a list of monthly variances."""
    arr = np.reshape(df[var].to_numpy(), (-1, 12))
    return np.var(arr, axis=0)

def yearly_variance(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns a list of yearly variances."""
    arr = np.reshape(df[var].to_numpy(), (-1, 12))
    return np.var(arr, axis=1)

def monthly_skewness(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns a list of monthly skewness."""
    arr = np.reshape(df[var].to_numpy(), (-1, 12))
    return scipy.stats.skew(arr, axis=0, bias=True)

def yearly_skewness(df: pd.DataFrame, var: str = "Durchfluss_m3s"):
    """Returns a list of yearly skewness."""
    arr = np.reshape(df[var].to_numpy(), (-1, 12))
    return scipy.stats.skew(arr, axis=1, bias=True)

# -----------------------------------------
#           Trend analysis
# -----------------------------------------

def __preprocess(df: pd.DataFrame, which: str):
    if which == "monthly":
        x = df["Durchfluss_m3s"].tolist()
        t = np.arange(hyd_years(df)[0], hyd_years(df)[-1]+1, 1/12)
        n = len(t)
    elif which == "yearly":
        x = yearly_mean(df)
        t = np.arange(hyd_years(df)[0], hyd_years(df)[-1]+1, 1)
        n = len(t)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")
    
    return x, t, n

def linreg(df: pd.DataFrame, which: str):
    """Returns slope, intercept, r, p, std_err of the linear 
    regression model using scipy.stats.linregress.
    
    Alternativ:
    x, t, n = __preprocess(df, which)
    mean_x = np.mean(x)
    mean_t = np.mean(t)
    sum_xt = sum([x_i * t_i for x_i, t_i in zip(x, t)])
    sum_t2 = sum(t_i**2 for t_i in t)
    slope = (sum_xt - n*mean_x*mean_t) / (sum_t2 - n*mean_t**2)
    intercept = mean_x - slope*mean_t
    Linreg = namedtuple("LineareRegression", ["slope", "intercept"])
    return Linreg(slope, intercept)
    """

    x, t, _ = __preprocess(df, which)
    return scipy.stats.linregress(t, x, alternative="two-sided")

def t_test_statistic(df: pd.DataFrame, which: str):
    """Returns test statistic for the t-test."""
    return linreg(df, which=which).slope / linreg(df, which=which).stderr

def mk_test(df: pd.DataFrame, which: str):
    if which == "monthly":
        return mk.seasonal_test(df["Durchfluss_m3s"].to_numpy(), alpha=0.05, period=12)
    elif which == "yearly":
        return mk.original_test(yearly_mean(df), alpha=0.05)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")

def moving_average(df: pd.DataFrame, which: str, window: int):
    """Returns the moving average of the time series."""
    x, _, _ = __preprocess(df, which)
    return np.convolve(x, np.ones(window), "valid") / window

def detrend_signal(df: pd.DataFrame) -> None:
    """Detrend the time series."""
    df["trendber"] = np.mean(df["Durchfluss_m3s"].to_numpy()) + \
        scipy.signal.detrend(df["Durchfluss_m3s"].to_numpy(), type="linear")
    df["trend"] = df["Durchfluss_m3s"].to_numpy() - df["trendber"].to_numpy()
    return df

# -----------------------------------------
#            Seasonal analysis
# -----------------------------------------

def calc_spectrum(df: pd.DataFrame, sample_rate: int = 1) -> None:
    """Calculate the FFT of the time series."""

    data = df["Durchfluss_m3s"].to_numpy()
    n = len(data)

    # Detrend signal
    data = np.mean(data) + scipy.signal.detrend(data, type="linear")
    
    # Apply window function (tapering)
    # blackman, hamming, flattop, tukey, cosine, boxcar
    data = scipy.signal.windows.cosine(M=n) * data
    
    # 1D Discrete Fourier Transform
    fft_output = scipy.fft.fft(data)
    
    # Remove first element (mean) and frequencies above Nyquist frequency.
    fft_output = fft_output[1:n//2]
    
    # Discrete Fourier Transform sample frequencies
    freqs = scipy.fft.fftfreq(n, 1/sample_rate)[1:n//2]
    
    # Calculate the square of the norm of each complex number
    # Norm = sqrt(Re² + Im²)
    spectrum = np.square(np.abs(fft_output))

    # Multiply spectral energy density by frequency
    # spectrum *= freqs
    
    # Multiply spectrum by 2 to account for negative frequencies
    spectrum = [i*2 for i in spectrum]
    
    return freqs, spectrum

def get_dominant_frequency(
        freqs: np.ndarray, 
        spectrum: np.ndarray, 
        n: int
        ):
    """Returns the n-th most dominant frequencies in ascending order."""
    idx = np.argpartition(spectrum, -n)[-n:]
    freqs = freqs[idx] # unit: 1/month
    period = [round(1/f, 2) for f in freqs] # unit: months
    
    return freqs, period

def season_signal(df: pd.DataFrame):
    saisonfigur_mean = np.tile(monthly_mean(df), 40)
    saisonfigur_var = np.tile(monthly_variance(df), 40)
    saisonfigur_std = np.sqrt(saisonfigur_var)
    df["saisonfigur_mean"] = saisonfigur_mean
    df["saisonfigur_std"] = saisonfigur_std
    df["saisonber"] = df["trendber"] - df["saisonfigur_mean"]
    return df

# -----------------------------------------
#           autocorrelation
# -----------------------------------------

def autocorr(df: pd.DataFrame, lag: int, var: str = "Durchfluss_m3s"):
    """Returns the autocorrelation."""
    return df[var].autocorr(lag=lag)

def confidence_interval(df: pd.DataFrame):
    """Returns the confidence interval."""
    lower_conf = -1.96 / np.sqrt(len(df))
    upper_conf = 1.96 / np.sqrt(len(df))
    return lower_conf, upper_conf

def monthly_autocorr(df: pd.DataFrame, lag: int, var: str = "Durchfluss_m3s"):
    """Returns a list of monthly autocorrelations."""
    # TODO

def yearly_autocorr(
    df: pd.DataFrame,
    lag: int,
    var: str = "Durchfluss_m3s"):
    """Returns a list of yearly autocorrelations."""
    arr = np.reshape(df[var].to_numpy(), (-1, 12))
    return [pd.Series(i).autocorr(lag=lag) for i in arr]
