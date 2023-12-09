import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import sys
sys.path.insert(1, '/home/lena/Doconfigkumente/FGB/StochastischeSpeicherbemessung/')

from settings import var_remapper
from utils.stats import *


# TU CD colors
tu_darkblue = (0/255, 48/255, 93/255)
tu_mediumblue = (0/255, 105/255, 180/255)
tu_grey = (114/255, 119/255, 119/255)
tu_red = (181/255, 28/255, 28/255)

MONTH_ABB = ["N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S", "O"]


def plot_raw(df: pd.DataFrame, fn: str) -> None:
    """Plot raw data."""
        
    max_value = max_val(df)[0]
    max_month = max_val(df)[1]
    min_value = min_val(df)[0]
    min_month = min_val(df)[1]

    max_month_date = df["Datum"][df["Monat"] == max_month].iloc[0]
    min_month_date = df["Datum"][df["Monat"] == min_month].iloc[0]

    plt.figure(figsize=(10, 4))
    plt.plot(df["Datum"], df["Durchfluss_m3s"], 
                c=tu_mediumblue, linewidth=0.8, label="Rohdaten")
    plt.axhline(y=max_value, c=tu_red, linestyle="--", linewidth=0.8, 
                label=f"Max: {max_month}: {max_value} m³/s")
    plt.axhline(y=min_value, c=tu_grey, linestyle="--", linewidth=0.8, 
                label=f"Min: {min_month}: {min_value} m³/s")
    plt.scatter(max_month_date, max_value, marker="o", 
                facecolors='none', edgecolors=tu_red, s=30)
    plt.scatter(min_month_date, min_value, marker="o", 
                facecolors='none', edgecolors=tu_grey, s=30)
    plt.xlabel("Monat")
    plt.ylabel("Durchfluss [m³/s]")
    plt.xticks(df["Datum"][::24], df["Monat"][::24], rotation=90)
    plt.yticks(np.arange(0, max_value, 1), minor=False)
    plt.yticks(np.arange(0, max_value, 0.25), minor=True)
    plt.grid(which="major", axis="x", color="grey", alpha=0.05)
    plt.grid(which="major", axis="y", color="grey", alpha=0.40)
    plt.grid(which="minor", axis="y", color="grey", alpha=0.05)
    # plt.ylim(bottom=0)
    plt.xlim(left=df["Datum"].min(), right=df["Datum"].max())
    plt.legend(loc="upper right")
    plt.savefig(fn, dpi=300, bbox_inches="tight")

def plot_dsk(test_cumsum: np.ndarray, ref_cumsum: np.ndarray, fn: str) -> None:
    """Plot double sum curve."""
    
    plt.figure(figsize=(5,5))    
    for i in range(0, len(ref_cumsum), 12):
        plt.scatter(ref_cumsum[i], test_cumsum[i], color=tu_mediumblue, s=15,
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
    plt.savefig(fn, dpi=300, bbox_inches="tight")

def plot_breakpoint(df: pd.DataFrame, res, fn: str) -> None:
    """Plot breakpoint analysis."""
    
    mean_1, mean_2 = res.avg
    cp = df["Datum"].iloc[res.cp]
    
    plt.figure(figsize=(10, 4))

    # Plot raw data
    plt.plot(df["Datum"], df["Durchfluss_m3s"], alpha=0.4,
                c=tu_mediumblue, linewidth=0.8, label="Rohdaten")
    
    # Plot vertical line at breakpoint
    plt.axvline(x=cp, c="green", linestyle="--", linewidth=0.8,
                label=f"Breakpoint: {df['Monat'].iloc[res.cp]}")
    
    # Plot mean before breakpoint
    x = df["Datum"].iloc[0:res.cp]
    y = mean_1 * np.ones(len(x))
    plt.plot(x, y, c=tu_red, linestyle="--", linewidth=0.8)
    plt.scatter(df["Datum"].iloc[0], mean_1, marker="o",
                facecolors='none', edgecolors=tu_red, s=30)
    plt.scatter(cp, mean_1, marker="o",
                facecolors='none', edgecolors=tu_red, s=30)
    plt.text(df["Datum"].iloc[int(res.cp/2)], mean_1+0.05, f"{mean_1:.2f}", ha="center", 
             va="bottom", fontsize=8, color=tu_red)
    
    # Plot mean after breakpoint
    x = df["Datum"].iloc[res.cp:]
    y = mean_2 * np.ones(len(x))
    plt.plot(x, y, c=tu_red, linestyle="--", linewidth=0.8, label="Mittelwert")
    plt.scatter(cp, mean_2, marker="o",
                facecolors='none', edgecolors=tu_red, s=30)
    plt.scatter(df["Datum"].iloc[-1], mean_2, marker="o",
                facecolors='none', edgecolors=tu_red, s=30)
    plt.text(df["Datum"].iloc[int((len(df)+res.cp)/2)], mean_2+0.05, f"{mean_2:.2f}", ha="center",
                va="bottom", fontsize=8, color=tu_red)
    
    plt.ylabel("Durchfluss [m³/s]")
    plt.xticks(df["Datum"][::24], df["Monat"][::24], rotation=90)
    plt.yticks(np.arange(0, 8, 1), minor=False)
    plt.yticks(np.arange(0, 8, 0.25), minor=True)
    plt.grid(which="major", axis="x", color="grey", alpha=0.05)
    plt.grid(which="major", axis="y", color="grey", alpha=0.40)
    plt.grid(which="minor", axis="y", color="grey", alpha=0.05)
    plt.legend(loc="upper right")
    plt.savefig(fn, dpi=300, bbox_inches="tight")

def plot_trend(
        df: pd.DataFrame,
        linreg_m: np.ndarray,
        linreg_y: np.ndarray,
        mk_m: np.ndarray, 
        mk_y: np.ndarray,
        ma_m: np.ndarray,
        ma_y: np.ndarray,
        fn: str
        ) -> None:
    """Plot trend analysis summary."""
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
    
    years = hyd_years(df)
    x_yearly = years
    x_monthly = np.arange(years[0], years[-1]+1, 1/12)
    
    # raw data
    ax1.plot(df["Datum"], df["Durchfluss_m3s"], c=tu_grey, linewidth=0.8, 
             alpha=0.3, marker="o", markersize=2, label="Monatswerte (Rohdaten)")
    ax2.plot(x_yearly, binned_stats(df, var="Durchfluss_m3s", bin="yearly", func=np.mean), c=tu_grey, linewidth=0.8, 
             alpha=0.3, marker="o", markersize=3, label="Jahreswerte (arith. Mittel)")
    
    # linear regression
    linreg_m_func = f"y = {linreg_m.slope:.5f}x + {linreg_m.intercept:.5f}"
    linreg_y_func = f"y = {linreg_y.slope:.5f}x + {linreg_y.intercept:.5f}"

    ax1.plot(df["Datum"], linreg_m.intercept + linreg_m.slope*x_monthly, c=tu_red, ls="--",
                label=f"Lineare Regression {linreg_m_func}")
    ax2.plot(x_yearly, linreg_y.intercept + linreg_y.slope*x_yearly, c=tu_red, ls="--",
                label=f"Lineare Regression {linreg_y_func}")
    
    # theil sen regression
    mk_m_func = f"y = {mk_m.slope:.5f}x + {mk_m.intercept:.5f}"
    mk_y_func = f"y = {mk_y.slope:.5f}x + {mk_y.intercept:.5f}"
    
    ax1.plot(df["Datum"], mk_m.intercept + mk_m.slope*x_monthly, c="green", ls="--",
                label=f"Theil-Sen-Regression {mk_m_func}")
    ax2.plot(x_yearly, mk_y.intercept + mk_y.slope*x_yearly, c="green", ls="--",
                label=f"Theil-Sen-Regression {mk_y_func}")
    
    # moving average
    ax1.plot(df["Datum"][5:-6], ma_m, c=tu_mediumblue, lw=0.8,
                label=f"Gleitender Durchschnitt (Fensterbreite: 1a)")
    ax2.plot(x_yearly[2:-2], ma_y, c=tu_mediumblue, lw=0.8,
                label=f"Gleitender Durchschnitt (Fensterbreite: 5a)")
    
    ax1.set_ylim([0,8])
    ax1.set_xticks(df["Datum"][::24], minor=False)
    ax1.set_xticklabels(df["Monat"][::24], rotation=90)
    ax1.set_yticks(np.arange(0, 8.5, 0.5), minor=True)
    ax1.set_xlim(left=df["Datum"].min(), right=df["Datum"].max())
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
    plt.savefig(fn, dpi=300, bbox_inches="tight")

def plot_spectrum(
        freqs: np.ndarray,
        spectrum: np.ndarray,
        fn: str
        ) -> None:
    """Plot FFT spectrum"""
    
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
    plt.savefig(fn, dpi=300, bbox_inches="tight")

def plot_sin_waves(
        freqs: np.ndarray, 
        period: np.ndarray,
        fn: str
        ) -> None:
    """Plot the dominant frequencies as sin waves."""

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
    plt.savefig(fn, dpi=300, bbox_inches="tight")

    
def plot_acf(
        lags: list[float],
        ac: list[float],
        lower_conf: list[float], 
        upper_conf: list[float],
        fn: str
        ) -> None:
    """Plots the autocorrelation function."""
    
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
    plt.savefig(fn, dpi=300, bbox_inches="tight")

def plot_characteristics(df: pd.DataFrame, fn: str) -> None:
    """Plot statistics and histograms for different components."""
    
    _, ax = plt.subplots(nrows=3, ncols=4, figsize=(13, 10))
    labels = ["Rohdaten", "Zufallskomponente"]
    vars = ["Durchfluss_m3s", "zufall"]
    colors = [tu_grey, "orange"]
    ylabels = ["Arith. Mittel (m³/s)", "Standardabweichung m³/s", "Schiefe (-)", "Autokorrelation $r_{1}$(-)"]
    letters = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J.", "K.", "L."]
    
    for i in range(len(labels)):
        var, label, color = vars[i], labels[i], colors[i]

        x_months = np.arange(1, 13)
        x_years = hyd_years(df)
        
        # upper row plots
        ax[0,0].plot(x_months, binned_stats(df, var=var, bin="monthly", func=np.mean),
                    c=color, linewidth=1, label=label)
        ax[0,1].plot(x_months, binned_stats(df, var=var, bin="monthly", func=np.std),
                    c=color, linewidth=1, label=label)
        ax[0,2].plot(x_months, binned_stats(df, var=var, bin="monthly", func=scipy.stats.skew),
                    c=color, linewidth=1, label=label)
        ax[0,3].plot(x_months, monthly_autocorr(df, var=var, which="pearson"),
                    c=color, linewidth=1, label=label)
        
        # middle row plots
        ax[1,0].plot(x_years, binned_stats(df, var=var, bin="yearly", func=np.mean),
                    c=color, linewidth=1, label=label)
        ax[1,1].plot(x_years, binned_stats(df, var=var, bin="yearly", func=np.std),
                    c=color, linewidth=1, label=label)
        ax[1,2].plot(x_years, binned_stats(df, var=var, bin="yearly", func=scipy.stats.skew),
                    c=color, linewidth=1, label=label)
        ax[1,3].plot(x_years, yearly_autocorr(df, lag=1, var=var),
                    c=color, linewidth=1, label=label)
        
        # lower row plots
        hist_kwargs = {"bins": np.arange(-2, 8, 0.25), "lw": 0.8, "edgecolor": "black", "alpha": 0.8}
        ax[2,i].hist(df[vars[i]], color=colors[i], label=labels[i], **hist_kwargs)
    
    for row_i in range(len(ax)):
        for col_i in range(len(ax[row_i])):
            ax[row_i, col_i].set_title(letters[row_i*len(ax[row_i])+col_i], loc='left', fontsize=12, fontweight="bold")
            ax[row_i, col_i].grid()
            ax[0, col_i].set_xlabel("Monat")
            ax[1, col_i].set_xlabel("Jahre")
            ax[2, col_i].set_xlabel("Durchfluss [m³/s]")
            ax[2, col_i].set_ylabel("Emp. Häufigkeit")
            ax[0, col_i].set_ylabel(ylabels[col_i])
            ax[1, col_i].set_ylabel(ylabels[col_i])
            ax[0, col_i].set_xticks(x_months)
            ax[0, col_i].set_xticklabels(MONTH_ABB)
            ax[1, col_i].set_xticks([1960, 1970, 1980, 1990, 2000])
            ax[1, col_i].set_xticklabels([1960, 1970, 1980, 1990, 2000])
            ax[2, col_i].set_ylim([0, 130])
            ax[2, 0].legend(frameon=False, loc="upper left")
            ax[2, 1].legend(frameon=False, loc="upper left")
            
    ax[2,2].set_visible(False)
    ax[2,3].set_visible(False)
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")    

def plot_characteristics_short(df: pd.DataFrame, fn: str) -> None:
    """Plot statistics and histograms for different components."""
    
    _, ax = plt.subplots(nrows=3, ncols=2, figsize=(13, 10))
    labels = ["Rohdaten", "Zufallskomponente"]
    vars = ["Durchfluss_m3s", "zufall"]
    colors = [tu_grey, "orange"]
    ylabels = ["Arith. Mittel (m³/s)", "Standardabweichung m³/s"]
    letters = ["A. Arith. Mittel", "B. Standardabweichung", "C. Arith. Mittel", "D. Standardabweichung", "E. Emp. Häufigkeit", "F. Emp. Häufigkeit"]
    
    for i in range(len(labels)):
        var, label, color = vars[i], labels[i], colors[i]

        x_months = np.arange(1, 13)
        x_years = hyd_years(df)
        
        # upper row plots
        ax[0,0].plot(x_months, binned_stats(df, var=var, bin="monthly", func=np.mean),
                    c=color, linewidth=1, label=label)
        ax[0,1].plot(x_months, binned_stats(df, var=var, bin="monthly", func=np.std),
                    c=color, linewidth=1, label=label)
         
        # middle row plots
        ax[1,0].plot(x_years, binned_stats(df, var=var, bin="yearly", func=np.mean),
                    c=color, linewidth=1, label=label)
        ax[1,1].plot(x_years, binned_stats(df, var=var, bin="yearly", func=np.std),
                    c=color, linewidth=1, label=label)
        
        # lower row plots
        hist_kwargs = {"bins": np.arange(-2, 8, 0.25), "lw": 0.8, "edgecolor": "black", "alpha": 0.8}
        ax[2,i].hist(df[vars[i]], color=colors[i], label=labels[i], **hist_kwargs)
    
    for row_i in range(len(ax)):
        for col_i in range(len(ax[row_i])):
            ax[row_i, col_i].set_title(letters[row_i*len(ax[row_i])+col_i], loc='left', fontsize=12, fontweight="bold", color="grey")
            ax[row_i, col_i].grid()
            ax[0, col_i].set_xlabel("Monat")
            ax[1, col_i].set_xlabel("Jahre")
            ax[2, col_i].set_xlabel("Durchfluss [m³/s]")
            ax[2, col_i].set_ylabel("Emp. Häufigkeit")
            ax[0, col_i].set_ylabel(ylabels[col_i])
            ax[1, col_i].set_ylabel(ylabels[col_i])
            ax[0, col_i].set_xticks(x_months)
            ax[0, col_i].set_xticklabels(MONTH_ABB)
            ax[1, col_i].set_xticks([1960, 1970, 1980, 1990, 2000])
            ax[1, col_i].set_xticklabels([1960, 1970, 1980, 1990, 2000])
            ax[2, col_i].set_ylim([0, 130])
            ax[2, col_i].legend(frameon=False, loc="upper left")
            
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")    

def pairplot(df: pd.DataFrame, fn: str) -> None:
    """Plot pairplot."""
    
    df = df.rename(columns=var_remapper)
    df["Monat"] = df["Monat"].str[:2].astype(int)
    df["Saison"] = df["Monat"].apply(
        lambda x: "Winter" if x in [12, 1, 2] else
        "Frühling" if x in [3, 4, 5] else
        "Sommer" if x in [6, 7, 8] else "Herbst")
    sns.pairplot(df, height=2.6, hue="Saison", palette ="bright", 
                 vars=[
                     "Rohdaten",
                     "Saisonale Komponente (Mittel)",
                     "Autokorrelative Komponente",
                     "Zufallskomponente"
                 ],
                 corner=False, plot_kws={"s":3, "alpha":0.5})
    plt.legend(fontsize='x-large', title_fontsize='50')
    plt.savefig(fn, dpi=300, bbox_inches="tight")

    
def plot_components(df: pd.DataFrame, fn: str) -> None:
    """Plot time series components."""

    _, ax = plt.subplots(nrows=9, ncols=1, figsize=(10,15), 
                           sharex=True, sharey=False)
    plot_config = {
        "Durchfluss_m3s": ("A. Rohdaten", tu_grey, [0, 8]),
        "trend": ("B. Trendkomponente (nicht sig.)", tu_red, [-0.1, 0.1]),
        "saisonfigur_mean": ("C. Saisonfigur (Mittelwert)", tu_mediumblue, [0, 4]),
        "saisonfigur_std": ("D.Saisonfigur (Standardabweichung)", tu_mediumblue, [0, 2]),
        "saisonber": ("E. Saisonbereinigte Zeitreihe", tu_mediumblue, [-2, 8]),
        "normiert": ("F. Normierte Zeitreihe", tu_mediumblue, [-2, 8]),
        "autokorr_saisonfigur": ("G. Autokorrelation (Saisonfigur)", "green", [-0.5, 1]),
        "autokorr": ("H. Autokorrelative Komponente", "green", [-2, 3]),
        "zufall": ("I. Zufallskomponente / Residuum", "orange", [-2, 5])
    }

    for i in range(len(plot_config)):
        key = list(plot_config.keys())[i]
        title, color, ylim = plot_config[key]
        ax[i].plot(df["Datum"], df[key], c=color, lw=1)
        ax[i].set_title(title, loc="left", fontsize=12, fontweight="bold", color="grey")
        ax[i].set_ylim(ylim)
        ax[i].set_xticks(df["Datum"][::24])
        ax[i].set_xticklabels(df["Monat"][::24], rotation=90)
        ax[i].set_xlim(left=df["Datum"].min(), right=df["Datum"].max())
    
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")

