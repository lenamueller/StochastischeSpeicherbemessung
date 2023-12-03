import pandas as pd
import numpy as np
import scipy

from utils.plotting import plot_sin_waves, plot_spectrum
from utils.statistics import binned_stats


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

def seasonal_comp(df: pd.DataFrame):
    print("\n--------------------------------------")
    print("\nBestimmung der saisonalen Komponente\n")
    
    freqs, spectrum = calc_spectrum(df)
    plot_spectrum(freqs, spectrum)
    
    freqs, period = get_dominant_frequency(freqs, spectrum, n=5)
    print("Top 5 Frequenzen: ", freqs, "1/Monat")
    print("Top 5 Periodendauern", period, "Monate")
    plot_sin_waves(freqs, period)
    
    # calculate components
    df["saisonfigur_mean"] = np.tile(binned_stats(df, var="Durchfluss_m3s", bin="monthly", func=np.mean), 40)
    df["saisonfigur_std"] = np.tile(binned_stats(df, var="Durchfluss_m3s", bin="monthly", func=np.std), 40)
    df["saisonber"] = df["Durchfluss_m3s"] - df["saisonfigur_mean"]
    df["normiert"] = (df["Durchfluss_m3s"] - df["saisonfigur_mean"]) / df["saisonfigur_std"]

