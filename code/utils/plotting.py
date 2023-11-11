import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymannkendall as mk
import scipy

from config import image_path, pegelname, tu_mediumblue, tu_grey, tu_red
from utils.statistics import hyd_years, max_val, max_val_month, min_val, \
    min_val_month, hyd_years, monthly_autocorr, yearly_autocorr, \
    monthly_mean, yearly_mean, monthly_variance, yearly_variance, \
    monthly_skewness, yearly_skewness, confidence_interval, \
        linreg, moving_average, calc_spectrum, get_dominant_frequency


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
    plt.grid(which="major", axis="x", color="grey", alpha=0.05)
    plt.grid(which="major", axis="y", color="grey", alpha=0.40)
    plt.grid(which="minor", axis="y", color="grey", alpha=0.05)
    plt.ylim(bottom=0)
    plt.xlim(left=df["Monat"].min(), right=df["Monat"].max())
    plt.legend(loc="upper right")
    plt.savefig(image_path+f"{pegelname}_raw.png", dpi=300, bbox_inches="tight")

def plot_dsk(test_cumsum: np.ndarray, ref_cumsum: np.ndarray):
    """Plot double sum curve."""
    plt.figure(figsize=(5,5))    
    for i in range(0, len(ref_cumsum), 12):
        plt.scatter(ref_cumsum[i], test_cumsum[i], c=tu_mediumblue, s=15,
                    marker="x", zorder=2, label="Messdaten", alpha=0.75)
    res = scipy.stats.linregress(ref_cumsum, test_cumsum)
    m = res.slope
    b = res.intercept
    plt.plot(ref_cumsum, m*ref_cumsum+b, c=tu_red, alpha=0.8, linewidth=1,
             label=f"lineare Regressionsgerade: y = {m:.3f}x + {b:.3f}", zorder=1)
    plt.grid(color="grey", alpha=0.3)
    plt.xlabel("Kumulierter Durchfluss [m³/s] der Testreihe")
    plt.ylabel("Kumulierter Durchfluss [m³/s] der Prüfreihe")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    # plt.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.15), frameon=False)
    plt.savefig(image_path+f"{pegelname}_dsk.png", dpi=300, bbox_inches="tight")

def plot_trend(df: pd.DataFrame):
    """Plot trend analysis summary."""
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
    
    years = hyd_years(df)
    x_yearly = years
    x_monthly = np. arange(years[0], years[-1]+1, 1/12)
    
    # raw data
    ax1.plot(df["Monat"], df["Durchfluss_m3s"], c=tu_grey, linewidth=0.8, 
             alpha=0.3, marker="o", markersize=2, label="Monatswerte (Rohdaten)")
    ax2.plot(x_yearly, yearly_mean(df), c=tu_grey, linewidth=0.8, 
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
    
    ax1.set_ylim([0,8])
    # ax1.set_xlabel("Zeit (Monate)")
    ax1.set_xticks(df["Monat"][::12], minor=False)
    ax1.set_xticklabels(df["Monat"][::12], rotation=90)
    ax1.set_yticks(np.arange(0, 8.5, 0.5), minor=True)
    ax1.set_xlim(left=df["Monat"].min(), right=df["Monat"].max())
    ax1.grid(which="major", axis="x", color="grey", alpha=0.05)
    ax1.grid(which="major", axis="y", color="grey", alpha=0.05)
    ax1.grid(which="minor", axis="y", color="grey", alpha=0.05)
    ax2.set_ylim([0,3])
    ax2.set_ylabel("Zeit (Jahre)")
    ax2.set_xticks(x_yearly, minor=True)
    ax2.set_yticks(np.arange(0, 3.1, 0.5), minor=True)
    ax2.set_yticks(np.arange(0, 4, 1), minor=False)
    ax2.set_xlim(left=x_yearly[0], right=x_yearly[-1])
    ax2.grid(which="major", axis="x", color="grey", alpha=0.05)
    ax2.grid(which="minor", axis="x", color="grey", alpha=0.05)
    ax2.grid(which="major", axis="y", color="grey", alpha=0.05)
    ax2.grid(which="minor", axis="y", color="grey", alpha=0.05)
    
    for ax in [ax1, ax2]:
        ax.set_ylabel("Durchfluss [m³/s]")
        ax.legend(loc="upper left")
        
    fig.tight_layout()
    plt.savefig(image_path+f"{pegelname}_trend.png", dpi=300, bbox_inches="tight")

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

def plot_hist(df: pd.DataFrame):
    """Plot histogram of raw data."""
    
    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4),
                           sharex=True, sharey=True)
    
    bins = np.arange(-2, 8, 0.25)

    ax[0].hist(df["Durchfluss_m3s"], bins=bins, color=tu_grey, lw=0.8, edgecolor="black", alpha=0.8)
    ax[1].hist(df["saisonber"], bins=bins, color=tu_mediumblue, lw=0.8, edgecolor="black", alpha=0.8)
    ax[2].hist(df["zufall"], bins=bins, color=tu_red, lw=0.8, edgecolor="black", alpha=0.8)
    
    ax[0].set_ylabel("empirische Häufigkeit")
    ax[0].set_xlabel("Durchfluss der Rohdaten [m³/s]")
    ax[1].set_xlabel("Saisonbereinigter Durchfluss [m³/s]")
    ax[2].set_xlabel("Zufallskomponente des Durchflusses [m³/s]")
    
    for i in range(len(ax)):
        ax[i].set_xticks(np.arange(-2, 8, 1), minor=False)
        ax[i].set_xticks(np.arange(-2, 8, 0.25), minor=True)
        ax[i].set_yticks(np.arange(0, 120, 10), minor=False)
        ax[i].set_yticks(np.arange(0, 122.5, 2.5), minor=True)
        ax[i].grid(which="minor", axis="y", color="grey", alpha=0.05)
        ax[i].grid(which="major", axis="y", color="grey", alpha=0.40)
        ax[i].grid(which="minor", axis="x", color="grey", alpha=0.05)
        ax[i].grid(which="major", axis="x", color="grey", alpha=0.40)
    
    plt.savefig(image_path+f"{pegelname}_hist.png", dpi=300, bbox_inches="tight")

def plot_characteristics(df: pd.DataFrame):
    """Plot monthly mean and median (Saisonfigur)."""
    
    _, ax = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
    
    labels = ["Rohdaten", "Saisonbereinigte Zeitreihe", "Zufallskomponente"]
    vars = ["Durchfluss_m3s", "saisonber", "zufall"]
    colors = [tu_grey, tu_mediumblue, tu_red]
    titles = ["Arith. Mittel (m³/s)", "Varianz (m³/s)²", "Schiefe (-)", "Autokorrelation $r_{1}$(-)"]

    for i in range(len(labels)):
        var, label, color = vars[i], labels[i], colors[i]

        x_months = np.arange(1, 13)
        x_years = hyd_years(df)
        
        ax[0,0].plot(x_months, monthly_mean(df, var=var),
                    c=color, linewidth=1, label=label)
        ax[0,1].plot(x_months, monthly_variance(df, var=var),
                    c=color, linewidth=1, label=label)
        ax[0,2].plot(x_months, monthly_skewness(df, var=var),
                    c=color, linewidth=1, label=label)
        ax[0,3].plot(x_months, monthly_autocorr(df, which="pearson"),
                    c=color, linewidth=1, label=label)
        
        ax[1,0].plot(x_years, yearly_mean(df, var=var),
                    c=color, linewidth=1, label=label)
        ax[1,1].plot(x_years, yearly_variance(df, var=var),
                    c=color, linewidth=1, label=label)
        ax[1,2].plot(x_years, yearly_skewness(df, var=var),
                    c=color, linewidth=1, label=label)
        ax[1,3].plot(x_years, yearly_autocorr(df, lag=1),
                    c=color, linewidth=1, label=label)
        
    for row_i in range(len(ax)):
        for col_i in range(len(ax[row_i])):
            ax[0, col_i].set_xlabel("Monat")
            ax[0, col_i].set_xticks(x_months)
            ax[0, col_i].set_xticklabels(["N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S", "O"])
            ax[1, col_i].set_xlabel("Jahre")
            ax[1, col_i].set_xticks([1960, 1970, 1980, 1990, 2000])
            ax[1, col_i].set_xticklabels([1960, 1970, 1980, 1990, 2000])
            ax[0, col_i].set_title(titles[col_i])
            ax[row_i, col_i].grid()
            ax[1,0].legend(bbox_to_anchor=(0.8, -0.3), ncol=1, frameon=False)

    plt.tight_layout()
    plt.savefig(image_path+f"{pegelname}_characteristics.png", dpi=300, bbox_inches="tight")    

def plot_acf(df: pd.DataFrame, var: str = "normiert"):
    """Plots the autocorrelation function."""
    lags = np.arange(0,51,1)
    ac = [df[var].autocorr(lag=i) for i in lags]
    lower_conf, upper_conf = confidence_interval(df, lags)
    
    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lags, ac, marker="o", markersize=3, color=tu_mediumblue, lw=1, zorder=1)
    ax.plot(lags, lower_conf, color=tu_red, linestyle="--", alpha=0.5)
    ax.plot(lags, upper_conf, color=tu_red, linestyle="--", alpha=0.5)
    for i in range(len(ac)):
        if ac[i] < lower_conf[i] or ac[i] > upper_conf[i]:
            ax.scatter(lags[i], ac[i], color=tu_red, marker="o", s=10, zorder=2)
            ax.text(lags[i]+0.2, ac[i]+0.05, lags[i], color=tu_red, rotation=0, ha="center", va="top", fontsize=5)
    
    ax.set_xlim([0, 50])
    ax.set_ylim([-0.5, 1])
    ax.set_xticks(np.arange(0, 51, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 1.1, 0.1), minor=True)
    ax.grid(which="major", axis="x", color="grey", alpha=0.40)
    ax.grid(which="minor", axis="x", color="grey", alpha=0.05)
    ax.grid(which="major", axis="y", color="grey", alpha=0.05)
    ax.grid(which="minor", axis="y", color="grey", alpha=0.05)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autokorrelation")
    if var == "normiert":
        fn = image_path+f"{pegelname}_acf.png"
    elif var == "Durchfluss_m3s":
        fn = image_path+f"{pegelname}_acf_raw.png"
    else:
        raise ValueError("var must be 'normiert' or 'Durchfluss_m3s'")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    
def plot_components(
        df: pd.DataFrame
    ):
    
    _, ax = plt.subplots(nrows=9, ncols=1, figsize=(10,15), 
                           sharex=True, sharey=False)

    plot_config = {
        "Durchfluss_m3s": ("A.", "Rohdaten", tu_grey, [0, 8]),
        "trend": ("B.", "Trendkomponente (nicht sig.)", tu_red, [-0.1, 0.1]),
        "saisonfigur_mean": ("C.", "Saisonfigur (Mittelwert)", tu_mediumblue, [0, 4]),
        "saisonfigur_std": ("D.", "Saisonfigur (Standardabweichung)", tu_mediumblue, [0, 2]),
        "saisonber": ("E.", "Saisonbereinigte Zeitreihe", tu_mediumblue, [-2, 8]),
        "normiert": ("F.", "Normierte Zeitreihe", tu_mediumblue, [-2, 8]),
        "autokorr_saisonfigur": ("G.", "Autokorrelation (Saisonfigur)", "green", [-0.5, 1]),
        "autokorr": ("H.", "Autokorrelative Komponente", "green", [-2, 3]),
        "zufall": ("I.", "Zufallskomponente / Residuum", "green", [-2, 5])
    }

    for i in range(len(plot_config)):
        key = list(plot_config.keys())[i]
        letter, title, color, ylim = plot_config[key]
        ax[i].plot(df["Monat"], df[key], c=color, lw=1)
        ax[i].set_title(title)
        ax[i].text(0.01, 0.85, letter, transform=ax[i].transAxes, 
                fontsize=12, fontweight="bold")    
        ax[i].set_ylim(ylim)
        ax[i].set_xticks(df["Monat"][::12])
        ax[i].set_xticklabels(df["Monat"][::12], rotation=90)
        ax[i].set_xlim(left=df["Monat"].min(), right=df["Monat"].max())
        # TODO: ylabels
    
    plt.tight_layout()
    plt.savefig(image_path+f"{pegelname}_components.png", dpi=300, bbox_inches="tight")
    