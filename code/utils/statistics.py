import scipy
import numpy as np
import pandas as pd
import pymannkendall as mk
import pyhomogeneity as hg
from statsmodels.tsa.stattools import adfuller
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

# -----------------------------------------
#         Quality investigation
# -----------------------------------------

def outlier_test_iqr(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> tuple[float, float, pd.DataFrame]:
    """Returns a list of outliers using the IQR method."""
    q1 = df[var].quantile(q=0.25, interpolation="nearest")
    q3 = df[var].quantile(q=0.75, interpolation="nearest")
    iqr = q3 - q1
    g_upper = q3 + 1.5*iqr
    g_lower = q1 - 1.5*iqr
    return g_upper, g_lower, df.loc[(df[var] < g_lower) | (df[var] > g_upper)]

def outlier_test_zscore(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> tuple[float, float, pd.DataFrame]:
    """Returns a list of outliers using the z-score method."""
    g_upper = df[var].mean() + 3*df[var].std()
    g_lower = df[var].mean() - 3*df[var].std()
    return g_upper, g_lower, df.loc[(df[var] < g_lower) | (df[var] > g_upper)]

def outlier_test_grubbs(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> tuple[float, pd.DataFrame]:
    """Returns a list of outliers using the Grubbs method."""
    max_diff = np.max(np.abs(df[var] - df[var].mean()))
    s = np.std(df[var])
    g = max_diff / s
    return g, df.loc[df[var] > g]

def double_sum(test_gauge: list[float], ref_gauge: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Returns a list of double sums."""
    sum_test = np.cumsum(test_gauge)
    sum_ref = np.cumsum(ref_gauge)
    return sum_test, sum_ref

def pettitt_test(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> FunctionType:
    """Test for inhomogeneity using the Pettitt test."""
    return hg.pettitt_test(x=df[var], alpha=0.05)

def adf_test(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> adfuller:
    """Test for stationarity using the Augmented Dickey-Fuller test."""
    return adfuller(df[var])

# -----------------------------------------
#           Trend analysis
# -----------------------------------------

def __preprocess(df: pd.DataFrame, which: str):
    if which == "monthly":
        x = df["Durchfluss_m3s"].tolist()
        t = np.arange(hyd_years(df)[0], hyd_years(df)[-1]+1, 1/12)
        n = len(t)
    elif which == "yearly":
        x = binned_stats(df, var="Durchfluss_m3s", bin="yearly", func=np.mean).tolist()
        t = np.arange(hyd_years(df)[0], hyd_years(df)[-1]+1, 1)
        n = len(t)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")
    
    return x, t, n

def linreg(df: pd.DataFrame, which: str) -> scipy.stats.linregress:
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

def mk_test(df: pd.DataFrame, which: str) -> FunctionType:
    """Trend test using the Mann-Kendall test."""
    if which == "monthly":
        return mk.seasonal_test(df["Durchfluss_m3s"].to_numpy(), alpha=0.05, period=12)
    elif which == "yearly":
        return mk.original_test(binned_stats(df, var="Durchfluss_m3s", bin="yearly", func=np.mean), alpha=0.05)
    else:
        raise ValueError("which must be 'monthly' or 'yearly'")

def moving_average(df: pd.DataFrame, which: str, window: int) -> np.ndarray[float]:
    """Returns the moving average of the time series."""
    x, _, _ = __preprocess(df, which)
    return np.convolve(x, np.ones(window), "valid") / window

# -----------------------------------------
#            Seasonal analysis
# -----------------------------------------

def calc_spectrum(df: pd.DataFrame, sample_rate: int = 1) -> tuple[list, list]:
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
        ) -> tuple[np.ndarray, np.ndarray]:
    """Returns the n-th most dominant frequencies in ascending order."""
    idx = np.argpartition(spectrum, -n)[-n:]
    freqs = freqs[idx] # unit: 1/month
    period = [round(1/f, 2) for f in freqs] # unit: months
    
    return freqs, period

# -----------------------------------------
#           autocorrelation
# -----------------------------------------

def autocorrelation(df: pd.DataFrame, var: str, lag: int = 1) -> float:
    """Returns the autocorrelation function."""
    return pd.Series(df[var]).autocorr(lag=lag)

def confidence_interval(df: pd.DataFrame, lags: list[float]) -> tuple[list[float], list[float]]:
    """Returns the confidence interval."""
    k = lags
    n = len(df)
    T_ALPHA = 1.645 # alpha = 0.05
    lower_conf = (-1 - T_ALPHA*np.sqrt(n-k-1)) / (n-k+1)
    upper_conf = (1 + T_ALPHA*np.sqrt(n-k-1)) / (n-k+1)
    return lower_conf, upper_conf

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
#           calculate components
# -----------------------------------------

def calc_components(df: pd.DataFrame, detrend: bool = False) -> None:
    """Calculates the components of the additive time series model."""
    
    # Trendkomponente
    df["trendber"] = np.mean(df["Durchfluss_m3s"].to_numpy()) + \
        scipy.signal.detrend(df["Durchfluss_m3s"].to_numpy(), type="linear")
    df["trend"] = df["Durchfluss_m3s"].to_numpy() - df["trendber"].to_numpy()
    
    # Saisonale Komponente
    df["saisonfigur_mean"] = np.tile(binned_stats(df, var="Durchfluss_m3s", bin="monthly", func=np.mean), 40)
    df["saisonfigur_std"] = np.tile(binned_stats(df, var="Durchfluss_m3s", bin="monthly", func=np.std), 40)
    if detrend:
        df["saisonber"] = df["trendber"] - df["saisonfigur_mean"]
        df["normiert"] = (df["trendber"] - df["saisonfigur_mean"]) / df["saisonfigur_std"]
    else:
        df["saisonber"] = df["Durchfluss_m3s"] - df["saisonfigur_mean"]
        df["normiert"] = (df["Durchfluss_m3s"] - df["saisonfigur_mean"]) / df["saisonfigur_std"]
    
    # Autokorrelative Komponente
    df["autokorr_saisonfigur"] = np.tile(monthly_autocorr(df=df), 40)
    df["autokorr"] = df["autokorr_saisonfigur"] * df["normiert"]
    
    # Zufallskomponente
    df["zufall"] = df["normiert"] - df["autokorr"]
    
    for var in ["trendber", "trend", "saisonfigur_mean", "saisonfigur_std",
                "saisonber", "normiert", "autokorr_saisonfigur", "autokorr",
                "zufall"]:
        df[var] = round(df[var], 5)
    
    