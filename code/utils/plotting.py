import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from setup import tu_darkblue, tu_mediumblue, tu_grey, tu_red # TU colors
from setup import image_path, pegelname
from utils.primary_stats import max_val, max_val_month, min_val, min_val_month, hyd_years
from utils.trend_analysis import linreg_monthly, linreg_yearly
from utils.fft_analysis import calc_spectrum, get_dominant_frequency
from utils.binned_stats import mean, median, variance, skewness
from utils.data_structures import df_to_np

def check_path(path):
    """Check if path exists, otherwise create it."""
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def plot_raw(df: pd.DataFrame):
    """Plot raw data."""
    
    check_path(image_path)
        
    max_value = max_val(df)
    max_month = max_val_month(df)
    min_value = min_val(df)
    min_month = min_val_month(df)

    plt.figure(figsize=(10, 5))
    plt.plot(df["Monat"], df["Durchfluss_m3s"], 
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
    plt.xticks(df["Monat"][::12], rotation=90)
    plt.yticks(np.arange(0, max_value, 1), minor=False)
    plt.yticks(np.arange(0, max_value, 0.25), minor=True)
    plt.grid(which="major", axis="x", color="grey", alpha=0.15)
    plt.grid(which="major", axis="y", color="grey", alpha=0.75)
    plt.grid(which="minor", axis="y", color="grey", alpha=0.15)
    plt.ylim(bottom=0)
    plt.xlim(left=df["Monat"].min(), right=df["Monat"].max())
    plt.legend(loc="upper right")
    
    plt.savefig(image_path+f"{pegelname}_raw.png", dpi=300, bbox_inches="tight")
    return None

def plot_hist(df: pd.DataFrame):
    """Plot histogram of raw data."""
    check_path(image_path)
    
        
    max_value = max_val(df)

    plt.figure(figsize=(10, 5))
    plt.hist(df["Durchfluss_m3s"], bins=np.arange(0, max_value+0.1, 0.1), 
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
    
    plt.savefig(image_path+f"{pegelname}_hist.png", dpi=300, bbox_inches="tight")
    return None

def plot_trend(df: pd.DataFrame):
    # todo
    
    check_path(image_path)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    
    days = np.arange(1, len(df)+1, 1)
    res_m = linreg_monthly(df)
    res_y = linreg_yearly(df)
    
    # store regression function in str
    res_m_func = f"y = {res_m.slope:.4f}x + {res_m.intercept:.4f}"
    res_y_func = f"y = {res_y.slope:.4f}x + {res_y.intercept:.4f}"
    
    ax1.plot(df["Monat"], df["Durchfluss_m3s"], 
                c=tu_mediumblue, linewidth=0.8, label="Monatswerte (Rohdaten)")
    ax1.plot(days, res_m.intercept + res_m.slope*days, c=tu_red, 
                label=f"lin. Regressionsgerade {res_m_func} (R²: {res_m.rvalue**2:.3f})")
    
    ax2.plot(hyd_years(df), mean(df, which="yearly"), label="Jahreswerte (arith. Mittel)")
    ax2.plot(hyd_years(df), res_y.intercept + res_y.slope*hyd_years(df), c=tu_red,
                label=f"lin. Regressionsgerade {res_y_func} (R²: {res_y.rvalue**2:.3f})")
    
    ax1.set_ylim([0,8])
    ax1.set_xlabel("Zeit (Monate)")
    ax1.set_xticks(df["Monat"][::12], minor=False)
    ax1.set_xticklabels(df["Monat"][::12], rotation=90)
    ax1.set_yticks(np.arange(0, 8.5, 0.5), minor=True)
    ax1.set_xlim(left=df["Monat"].min(), right=df["Monat"].max())
    ax1.grid(which="major", axis="x", color="grey", alpha=0.15)
    ax1.grid(which="major", axis="y", color="grey", alpha=0.15)
    ax1.grid(which="minor", axis="y", color="grey", alpha=0.15)
        
    ax2.set_ylim([0,3])
    ax2.set_ylabel("Zeit (Jahre)")
    ax2.set_xticks(hyd_years(df), minor=True)
    ax2.set_yticks(np.arange(0, 3.1, 0.5), minor=True)
    ax2.set_yticks(np.arange(0, 4, 1), minor=False)
    ax2.set_xlim(left=hyd_years(df)[0], right=hyd_years(df)[-1])
    ax2.grid(which="major", axis="x", color="grey", alpha=0.15)
    ax2.grid(which="minor", axis="x", color="grey", alpha=0.15)
    ax2.grid(which="major", axis="y", color="grey", alpha=0.15)
    ax2.grid(which="minor", axis="y", color="grey", alpha=0.15)
    
    for ax in [ax1, ax2]:
        ax.set_ylabel("Durchfluss [m³/s]")
        ax.legend(loc="upper right")
        
    fig.tight_layout()
    
    plt.savefig(image_path+f"{pegelname}_trend.png", dpi=300, bbox_inches="tight")
    return None

def plot_spectrum(df: pd.DataFrame):
    """Plot FFT spectrum"""
    check_path(image_path)
    
    
    freqs, spectrum = calc_spectrum(df)
    
    _, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(freqs[1:], spectrum[1:], c=tu_mediumblue)
    ax2 = ax1.secondary_xaxis(-0.18)
    x_ticks_ax1 = [1/24, 1/12, 1/6, 1/4, 1/3, 1/2]
    x_ticks_ax1 = [round(i, 2) for i in x_ticks_ax1]
    x_ticks_ax2 = [24, 12, 6, 4, 3, 2]
    ax1.set_xticks(x_ticks_ax1)
    ax2.set_xticks(x_ticks_ax1)
    ax2.set_xticklabels(x_ticks_ax2)
    ax1.set_xlabel("Frequenz [1/Monat]")
    ax2.set_xlabel("Periode [Monate]")
    ax1.set_ylabel("Spektrale Dichte")
    plt.grid()
    
    plt.savefig(image_path+f"{pegelname}_fft.png", dpi=300, bbox_inches="tight")
    return None

def plot_sin_waves(df):
    """Plot the dominant frequencies as sin waves."""
    check_path(image_path)
    freqs, period = get_dominant_frequency(*calc_spectrum(df), n=5)
    freqs = freqs[:-1] # del mean
    period = period[:-1] # del mean
    x = np.linspace(0, 12*2*np.pi, 1000)
    colors = ["gold", "salmon", "red", "darkred"]
    
    _, ax = plt.subplots(figsize=(10, 5))
    
    # plot single sin waves        
    for i in range(len(freqs)):
        y = np.sin(freqs[i] * x)
        ax.plot(x, y, label=f"{round(period[i], 1)} Monate", 
                c=colors[i], linewidth=1.5)
    
    # plot sum sin wave
    y = np.zeros(1000)
    for i in range(len(freqs)):
        y += np.sin(freqs[i] * x)
    ax.plot(x, y, label="Summe", c=tu_mediumblue, linewidth=2, linestyle="--")
    
    x_ticks = np.linspace(0, 12*2*np.pi, 13)
    x_ticks_labels = [round(i/(2*np.pi), 1) for i in x_ticks]
    x_ticks_labels = [int(x+1) for x in x_ticks_labels]
    ax.set_xlim([0,12])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels)
    ax.set_xlabel("Zeit [Monate]")
    ax.set_ylabel("Amplitude")
    ax.grid()
    ax.legend()
    
    plt.savefig(image_path+f"{pegelname}_sin.png", dpi=300, bbox_inches="tight")

def plot_saisonfigur(df: pd.DataFrame):
    """Plot monthly mean and median (Saisonfigur)."""
    check_path(image_path)
    
    data_np = df_to_np(df)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, 13)
    
    ax.plot(x, mean(df, which="monthly"), 
            c="green", linewidth=1.5, label="Arith. Mittel")
    ax.plot(x, median(df, which="monthly"),
            c=tu_mediumblue, linewidth=1.5, label="Median")
    ax.plot(x, variance(df, which="monthly"), 
            c=tu_red, linewidth=1.5, label="Varianz")
    ax.plot(x, skewness(df, which="monthly"),
            c="orange", linewidth=1.5, label="Skewness")
    
    for i in range(len(data_np)):
        ax.plot(x, data_np[i, :], linewidth=0.5, alpha=0.3, c=tu_grey)
    ax.set_xlabel("Monat")
    ax.set_ylabel("Durchfluss [m³/s]")
    ax.set_xticks(x)
    ax.set_xticklabels(["Nov", "Dez", "Jan", "Feb", "Mär", "Apr", 
                        "Mai", "Jun", "Jul", "Aug", "Sep", "Okt"])
    ax.grid()
    plt.legend()
    plt.savefig(image_path+f"{pegelname}_saison.png", dpi=300, bbox_inches="tight")