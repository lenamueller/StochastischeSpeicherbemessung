import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import lognorm, gamma, skew
import sys
sys.path.insert(1, '/home/lena/Dokumente/FGB/StochastischeSpeicherbemessung/')

from settings import N_GEN_TIMESERIES
from utils.read import monthly_vals
from utils.stats import binned_stats, monthly_autocorr


# TU CD colors
tu_darkblue = (0/255, 48/255, 93/255)
tu_mediumblue = (0/255, 105/255, 180/255)
tu_grey = (114/255, 119/255, 119/255)
tu_red = (181/255, 28/255, 28/255)


def plot_monthly_fitting(df: pd.DataFrame, fn: str) -> None:
    """Plot the fitting of the distribution for each month.""" 

    months = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    months_text = ["November", "Dezember", "Januar", "Februar", "März", "April",
                    "Mai", "Juni", "Juli", "August", "September", "Oktober"]
    bins = np.arange(0, 10, 0.25)

    _, axs = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
    for i in range(12):
        m = months[i]
        row_i = i // 4
        col_i = i % 4
        
        axs[row_i, col_i].set_title(f"{months_text[i]}", loc="left", color="grey", fontsize=10, fontweight="bold")
        axs[row_i, col_i].hist(monthly_vals(df, m), bins=bins, density=True, label="Messdaten", alpha=0.3)
        
        # LogNorm single months
        shape, loc, scale = lognorm.fit(monthly_vals(df, m))
        axs[row_i, col_i].plot(
            bins, lognorm.pdf(bins, s=shape, loc=loc, scale=scale),
            label="LogNorm",
            color="red", ls="-", lw=1.2
            )
        # LogNorm all months
        shape_all_months, loc_all_months, scale_all_months = lognorm.fit(df["Durchfluss_m3s"])
        axs[row_i, col_i].plot(
            bins, lognorm.pdf(bins, s=shape_all_months, loc=loc_all_months, scale=scale_all_months),
            label="LogNorm (Alle Monate)",
            color="red", ls="--", lw=1.2
            )
        # Gamma single months
        a, floc, scale = gamma.fit(monthly_vals(df, m))
        axs[row_i, col_i].plot(
            bins, gamma.pdf(bins, a, floc, scale), 
            label="Gamma",
            color="green", ls="-", lw=1.2
            )
        # Gamma all months
        a_all_months, floc_all_months, scale_all_months = gamma.fit(df["Durchfluss_m3s"])
        axs[row_i, col_i].plot(
            bins, gamma.pdf(bins, a_all_months, floc_all_months, scale_all_months), 
            label="Gamma (Alle Monate)",
            color="green", ls="--", lw=1.2
            )
        
    axs[0, 0].legend()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()

def plot_thomasfierung_eval(raw_data: pd.DataFrame, gen_data: pd.DataFrame, fn: str):
    """Plot evaluation of generated data compared to original data."""
    
    _, axs = plt.subplots(3, 2, figsize=(12, 11), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    
    x = np.arange(0, 12)
    titles = ["A. Arith. Mittel", "B. Varianz", "C. Schiefe", 
              "D. Autokorrelation", "E. Emp. Verteilung"]
    title_kwargs = {"loc": "left", "color": "grey", "fontsize": 10, 
                    "fontweight": "bold"}
    bins=np.arange(0, 10, 0.25)
    
    # raw data
    axs[0,0].plot(x, binned_stats(raw_data, var="Durchfluss_m3s", bin="monthly", func=np.mean),
                color=tu_mediumblue, alpha=1, lw=1)
    axs[0,1].plot(x, binned_stats(raw_data, var="Durchfluss_m3s", bin="monthly", func=np.var),
                color=tu_mediumblue, alpha=1, lw=1)
    axs[1,0].plot(x, binned_stats(raw_data, var="Durchfluss_m3s", bin="monthly", func=skew),
                color=tu_mediumblue, alpha=1, lw=1)
    axs[1,1].plot(x, monthly_autocorr(raw_data, var="Durchfluss_m3s"),
                  color=tu_mediumblue, alpha=1, lw=1)
    axs[2,0].hist(raw_data["Durchfluss_m3s"].to_numpy(), bins=bins, 
                density=False, alpha=0.1, color=tu_mediumblue)
    
    # gen data
    means = []
    vars = []
    skews = []
    autokorr = []
    hists = []
    
    for i in range(N_GEN_TIMESERIES):
        gen_i = gen_data[f"G{str(i+1).zfill(3)}"].to_numpy().reshape(-1, 12)
    
        m = np.mean(gen_i, axis=0)
        v = np.var(gen_i, axis=0)
        s = skew(gen_i, axis=0)
        a = monthly_autocorr(gen_data, var=f"G{str(i+1).zfill(3)}")
        h = np.histogram(gen_i, bins=bins, density=False)
    
        means.append(m)
        vars.append(v)
        skews.append(s)
        autokorr.append(a)
        hists.append(h)
        
        # Plot only first 40 time series
        if i < 101:
            axs[0,0].plot(x, m, color=tu_red, alpha=0.03, lw=1)
            axs[0,1].plot(x, v, color=tu_red, alpha=0.03, lw=1)
            axs[1,0].plot(x, s, color=tu_red, alpha=0.03, lw=1)
            axs[1,1].plot(x, a, color=tu_red, alpha=0.03, lw=1)
            axs[2,0].hist(gen_i.ravel()[:len(raw_data)], bins=bins, 
                        histtype="step", density=False, alpha=0.1, color=tu_red)

    # gen data means    
    axs[0,0].plot(x, np.mean(means, axis=0), color=tu_red, alpha=1, lw=1)
    axs[0,1].plot(x, np.mean(vars, axis=0), color=tu_red, alpha=1, lw=1)
    axs[1,0].plot(x, np.mean(skews, axis=0), color=tu_red, alpha=1, lw=1)
    axs[1,1].plot(x, np.mean(autokorr, axis=0), color=tu_red, alpha=1, lw=1)

    # plotting config
    for i in range(5):
        row_i = i // 2
        col_i = i % 2
        
        axs[row_i, col_i].grid(color="grey", alpha=0.3)
        axs[row_i, col_i].set_title(titles[i], **title_kwargs)
        axs[row_i, col_i].plot([], [], color=tu_mediumblue, label="original")
        axs[row_i, col_i].plot([], [], color=tu_red, label="generiert")
        axs[row_i, col_i].legend(fontsize=9, loc="upper left")

        if i != 4:
            axs[row_i, col_i].set_xticks(x)
            axs[row_i, col_i].set_xticklabels(["N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S", "O"])
            axs[row_i, col_i].set_xlabel("Monat")
        else:
            axs[row_i, col_i].set_xlabel("Durchfluss [m³/s]")
    
    axs[2,1].set_visible(False)
    
    plt.savefig(fn, dpi=300, bbox_inches="tight")      
    plt.close()   