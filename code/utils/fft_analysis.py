import pandas as pd
import numpy as np
import scipy

from utils.binned_stats import mean


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
    # TODO: #5 check algorithm argpartition
    
    idx = np.argpartition(spectrum, -n)[-n:]
    freqs = freqs[idx] # unit: 1/month
    period = [round(1/f, 2) for f in freqs] # unit: months
    
    return freqs, period

def season_signal(df: pd.DataFrame):
    df_season = df.copy(deep=True)
    df_deseason = df.copy(deep=True)
    df_season["Durchfluss_m3s"] = np.tile(mean(df, which="monthly"), 40)
    df_deseason["Durchfluss_m3s"] = df["Durchfluss_m3s"] - df_season["Durchfluss_m3s"]
    return df_season, df_deseason
    