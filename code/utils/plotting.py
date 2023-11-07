import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymannkendall as mk

from config import tu_darkblue, tu_mediumblue, tu_grey, tu_red # TU colors
from config import image_path, pegelname
from utils.primary_stats import max_val, max_val_month, min_val, min_val_month, hyd_years
from utils.trend_analysis import linreg, moving_average
from utils.fft_analysis import calc_spectrum, get_dominant_frequency
from utils.binned_stats import mean, median, variance, skewness
from utils.data_structures import df_to_np


def plot_raw(df: pd.DataFrame):
    """Plot raw data."""
        
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
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
    
    years = hyd_years(df)
    x_yearly = years
    x_monthly = np. arange(years[0], years[-1]+1, 1/12)
    
    # raw data
    ax1.plot(df["Monat"], df["Durchfluss_m3s"], c=tu_grey, linewidth=0.8, 
             alpha=0.3, marker="o", markersize=2, label="Monatswerte (Rohdaten)")
    ax2.plot(x_yearly, mean(df, which="yearly"), c=tu_grey, linewidth=0.8, 
             alpha=0.3, marker="o", markersize=3, label="Jahreswerte (arith. Mittel)")
    
    # linear regression
    linreg_m = linreg(df, which="monthly")
    linreg_y = linreg(df, which="yearly")
    linreg_m_func = f"y = {linreg_m.slope:.5f}x + {linreg_m.intercept:.5f}"
    linreg_y_func = f"y = {linreg_y.slope:.5f}x + {linreg_y.intercept:.5f}"

    ax1.plot(df["Monat"], linreg_m.intercept + linreg_m.slope*x_monthly, c=tu_red, ls="--",
                label=f"Lineare Regression {linreg_m_func}")
    ax2.plot(x_yearly, linreg_y.intercept + linreg_y.slope*x_yearly, c=tu_red, ls="--",
                label=f"Lineare Regression {linreg_y_func}")
    
    # theil sen regression
    mk_m = mk.original_test(df["Durchfluss_m3s"], alpha=0.05)
    mk_y = mk.seasonal_test(df["Durchfluss_m3s"], alpha=0.05, period=12)
    mk_m_func = f"y = {mk_m.slope:.5f}x + {mk_m.intercept:.5f}"
    mk_y_func = f"y = {mk_y.slope:.5f}x + {mk_y.intercept:.5f}"
    
    ax1.plot(df["Monat"], mk_m.intercept + mk_m.slope*x_monthly, c="green", ls="--",
                label=f"Theil-Sen-Regression {mk_m_func}")
    ax2.plot(x_yearly, mk_y.intercept + mk_y.slope*x_yearly, c="green", ls="--",
                label=f"Theil-Sen-Regression {mk_y_func}")
    
    # moving average
    ma_m = moving_average(df, which="monthly", window=12)
    ma_y = moving_average(df, which="yearly", window=5)
    
    ax1.plot(df["Monat"][5:-6], ma_m, c=tu_mediumblue, lw=0.8,
                label=f"Gleitender Durchschnitt (Fensterbreite: 1a)")
    ax2.plot(x_yearly[2:-2], ma_y, c=tu_mediumblue, lw=0.8,
                label=f"Gleitender Durchschnitt (Fensterbreite: 5a)")
    
    # difference method
    # diff_m = np.diff(df["Durchfluss_m3s"])
    # diff_y = np.diff(mean(df, which="yearly"))
    
    # ax1.plot(df["Monat"][1:], diff_m, c="orange", lw=0.8,
    #             label=f"Differenzmethode")
    # ax2.plot(x_yearly[1:], diff_y, c="orange", lw=0.8,
    #             label=f"Differenzmethode")
    
    
    
    # ax1.set_ylim([0,8])
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
    ax2.set_xticks(x_yearly, minor=True)
    ax2.set_yticks(np.arange(0, 3.1, 0.5), minor=True)
    ax2.set_yticks(np.arange(0, 4, 1), minor=False)
    ax2.set_xlim(left=x_yearly[0], right=x_yearly[-1])
    ax2.grid(which="major", axis="x", color="grey", alpha=0.15)
    ax2.grid(which="minor", axis="x", color="grey", alpha=0.15)
    ax2.grid(which="major", axis="y", color="grey", alpha=0.15)
    ax2.grid(which="minor", axis="y", color="grey", alpha=0.15)
    
    for ax in [ax1, ax2]:
        ax.set_ylabel("Durchfluss [m³/s]")
        ax.legend(loc="upper left")
        
    fig.tight_layout()
    
    plt.savefig(image_path+f"{pegelname}_trend.png", dpi=300, bbox_inches="tight")
    return None

def plot_spectrum(df: pd.DataFrame):
    """Plot FFT spectrum"""
    
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