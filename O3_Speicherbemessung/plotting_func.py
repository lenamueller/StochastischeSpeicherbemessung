import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# TU CD colors
tu_darkblue = (0/255, 48/255, 93/255)
tu_mediumblue = (0/255, 105/255, 180/255)
tu_grey = (114/255, 119/255, 119/255)
tu_red = (181/255, 28/255, 28/255)


MONTH_ABB = ["N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S", "O"]
MONTH_HYD_YEAR = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9 ,10]
MONTH_HYD_YEAR_TXT = ["Nov", "Dez", "Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep" ,"Okt"]


def plot_monthly_discharge(df_dis: pd.DataFrame, fn: str) -> None:
    """Plot monthly discharge values."""
    
    plt.figure(figsize=(12, 5))
    for i in range(len(df_dis)):
        if i == 0:
            # original time series    
            plt.plot(np.arange(1,13), df_dis.iloc[:, i].to_numpy(), color=tu_mediumblue, 
                 alpha=0.7, linewidth=0.8)
        else:
            # generated time series
            plt.scatter(np.arange(1,13), df_dis.iloc[:, i].to_numpy(), color=tu_red, 
                 alpha=0.5, marker="x")
    
    plt.plot([], [], color=tu_mediumblue, label="original")
    plt.scatter([], [], marker="x", color=tu_red, label="generiert")
    plt.legend(loc="upper left", ncols=3)
    plt.grid(color="grey", alpha=0.3)
    plt.xticks(np.arange(1, 13), MONTH_ABB)
    plt.xlabel("Monat")
    plt.ylabel("Durchfluss [hm³]")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()

def plot_fsa(
        storage: np.ndarray,
        max_vals: list[float],
        max_indices: list[float],
        min_vals: list[float],
        min_indices: list[float],
        cap: float,
        cap_min_index: float, 
        cap_min: float,
        cap_max: float,
        fn: str
        ) -> None:
    """Plot FSA algortihm."""
    
    plt.figure(figsize=(15,8))
    plt.title(f"Maximalkapazität des Speichers: {round(cap, 3)} hm³",
              loc="left", color="grey", fontsize=10, fontweight="bold")
    
    plt.plot(storage, label="Speicherinhalt [hm³]", color=tu_mediumblue, zorder=1)
    
    plt.scatter(max_indices, max_vals, s=25, marker="o", color="red", label="Maxima der SDL", zorder=2)
    plt.scatter(min_indices, min_vals, s=25, marker="o", color="green", label="Minima der SDL", zorder=2)
    
    plt.axvline(x=cap_min_index, color="k", linestyle="--", linewidth=0.8)
    plt.axhline(y=cap_min, color="k", linestyle="--", linewidth=0.8)
    plt.axhline(y=cap_max, color="grey", linestyle="--", linewidth=0.8)
    
    plt.text(100, cap_min, f"{round(cap_min, 3)} hm³", ha="left", 
             va="bottom", fontsize=10, color="k")
    plt.text(100, cap_max, f"{round(cap_max, 3)} hm³", ha="left", 
             va="bottom", fontsize=10, color="k")
    
    plt.legend(loc="upper center", fontsize=10)
    plt.grid(color="grey", alpha=0.3)
    plt.xlabel("Zeit [Monate]")
    plt.ylabel("Speicherinhalt [hm³]")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()

def plot_pu(
        capacities_sort: list[float],
        pu_emp: list[float],
        pu_theo: list[float],
        cap_90: float,
        fn: str
        ) -> None:
    """Pu plot of theoretical and empirical Pu."""
    
    plt.figure(figsize=(5,5))
    x_lower_lim = 5*round(min(capacities_sort)/5)-5
    x_upper_lim = 5*round(max(capacities_sort)/5)+5
    x_text = (cap_90 + x_lower_lim)/2
    
    plt.plot(capacities_sort, pu_emp, color=tu_red, label="Emp. $P_u$ [-]", 
             marker="x", markersize=5, alpha=0.5)
    plt.plot(capacities_sort, pu_theo, color=tu_mediumblue, alpha=0.5,
             label="Theoretische $P_u$ der LogNV [-]")
    plt.plot([cap_90, cap_90], [0, 0.9], color=tu_mediumblue, linestyle="--", alpha=0.5)
    plt.plot([0, cap_90], [0.9, 0.9], color=tu_mediumblue, linestyle="--", alpha=0.5)
    plt.text(cap_90+0.1, 0.5, f"K = {round(cap_90, 3)} hm³", ha="left", 
             va="center", fontsize=12, color=tu_mediumblue, rotation=270)
    plt.text(x_text, 0.91, f"$P_u$ = 0.9", ha="center", fontsize=10, color=tu_mediumblue)
    
    plt.xlabel("Maximalkapazität des Speichers [h³]")
    plt.ylabel("$P_u$ [-]")
    plt.yticks(np.arange(0,1.1,0.1))
    plt.ylim([0, 1])
    plt.xlim([x_lower_lim, x_upper_lim])
    plt.grid(color="grey", alpha=0.3)
    plt.legend(loc="lower right")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()

def plot_qq(emp: list[float], theo: list[float], fn: str) -> None:
    """Quantile-Quantile plot of theoretical and empirical quantiles."""
    
    r_qq = np.corrcoef(emp, theo)[0][1]
    
    plt.figure(figsize=(5,5))
    plt.scatter(theo, emp, color=tu_red, marker="x", s=5)
    plt.plot([0,50], [0, 50], color="k", alpha=0.4)
    plt.text(10.5, 43.5, f"$r_{{qq}}$ = {round(r_qq, 3)}", ha="left", 
             va="center", fontsize=10, color=tu_red)
    plt.xlabel("Theoretische Quantile [hm³]")
    plt.ylabel("Empirische Quantile [hm³]")
    plt.xticks(np.arange(0,50,5))
    plt.yticks(np.arange(0,50,5))
    plt.xlim([10, 45])
    plt.ylim([10, 45])
    plt.grid(color="grey", alpha=0.3)
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()

def plot_capacities_hist(capacities: list[float], hist_cap: float, fn: str) -> None:
    """Plot histogram of all generated capacities."""
    
    plt.figure(figsize=(6, 5))
    bins = np.arange(10, 46, 1)
    plt.hist(capacities, bins=bins, color=tu_grey, alpha=0.5, edgecolor="black")
    plt.axvline(x=hist_cap, color=tu_darkblue, linestyle="--", linewidth=1)
    plt.text(hist_cap+0.5, 12, f"{round(hist_cap, 3)} hm³", ha="left",
                va="bottom", fontsize=10, color=tu_darkblue)
    plt.ylabel("Absolute Häufigkeit [-]")
    plt.xlabel("Kapazität [hm³]")
    plt.grid(color="grey", alpha=0.3)
    plt.xlim([10,45])
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()

def plot_deficit_overflow(
        deficit: np.ndarray, 
        overflow: np.ndarray,
        months: np.ndarray, 
        fn: str
        ) -> None:
    """Plot monthly distribution of deficit and overflow situations."""
    
    # Bin months with deficit and overflow
    deficit_months = {
        11:0, 12:0, 1:0, 2:0, 3:0, 4:0, 
        5:0,6:0, 7:0, 8:0, 9:0, 10:0
        }
    overflow_months = {
        11:0, 12:0, 1:0, 2:0, 3:0, 4:0, 
        5:0,6:0, 7:0, 8:0, 9:0, 10:0
        }
    
    for i in range(len(deficit)):
        mo = int(months[i][:2])
        if deficit[i] < 0:
            deficit_months[mo] += 1
        if overflow[i] > 0:
            overflow_months[mo] += 1
    
    # Plot as hist
    _, axs = plt.subplots(1, 2, figsize=(9, 4))

    axs[0].set_title("Leerlauf", loc="left", color="grey", fontsize=10, fontweight="bold")
    axs[1].set_title("Überlauf", loc="left", color="grey", fontsize=10, fontweight="bold")
    axs[0].barh(MONTH_HYD_YEAR, deficit_months.values(), color=tu_red, alpha=0.5)
    axs[1].barh(MONTH_HYD_YEAR, overflow_months.values(), color=tu_red, alpha=0.5)
    for m in MONTH_HYD_YEAR:
        axs[0].text(deficit_months[m]+0.5, m, f"{deficit_months[m]}", ha="left",
                    va="center", fontsize=10, color=tu_red)
        axs[1].text(overflow_months[m]+0.5, m, f"{overflow_months[m]}", ha="left",
                    va="center", fontsize=10, color=tu_red)
    
    axs[1].set_xlim([0, max(overflow_months.values())+5])
    axs[0].set_xlim([0, max(deficit_months.values())+5])

    for i in [0, 1]:
        axs[i].set_xlabel("Anzahl Monate [-]")
        axs[i].set_yticks(MONTH_HYD_YEAR)
        axs[i].set_yticklabels(MONTH_HYD_YEAR_TXT)
        
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()

def plot_storage_simulation(
        q_in: np.ndarray,
        q_out: np.ndarray,
        q_out_real: np.ndarray,
        storage: np.ndarray, 
        deficit: np.ndarray,
        overflow: np.ndarray,
        var: str, 
        cap: float,
        initial_storage: float,
        xticklabels: list[str],
        fn: str
        ) -> None:
    """Plot storage simulation with inflow, outflow, storage volume and deficit/overflow."""
    
    # Number of month with deficit and overflow
    n_deficit = len([i for i in deficit if i < 0])
    n_overflow = len([i for i in overflow if i > 0])
    
    _, ax = plt.subplots(nrows=6, ncols=1, figsize=(10, 12), 
                         sharex=True)
    plt.suptitle(f"Speichersimulation [Zeitreihe: {var}]")
    titles = ["A. Zufluss", "B. Sollabgabe", "C. Istabgabe", "D. Zufluss-Sollabgabe", 
              f"E. Speicherinhalt [Anfangsfüllung = {initial_storage} hm³, Maximale Kapazität = {cap} hm³]", 
              f"F. Defizit ({n_deficit} Monate)/ Überlauf ({n_overflow} Monate)"]
    
    x = np.arange(len(q_in))
    ax[0].plot(x, q_in, c=tu_mediumblue)
    ax[1].plot(x, q_out, c=tu_grey)
    ax[2].plot(x, q_out_real, c=tu_grey)
    ax[3].plot(x, q_in-q_out, c=tu_red)
    ax[4].plot(x, storage, c="green")
    ax[5].bar(x, deficit, label="Defizit [hm³]")
    ax[5].bar(x, overflow, label="Überlauf [hm³]")

    for i in [0,1,2,3,5]:
        ax[i].set_ylabel("Volumen [hm³]")
    ax[4].set_ylabel("Volumen [hm³]")

    for i in range(6):
        ax[i].set_title(titles[i], loc="left", color="grey", fontsize=10, fontweight="bold")
        ax[i].set_xlabel("Zeit [Monate]")    
        ax[i].grid()
        
        # tick labels
        x_ticks = [x[0]] + list(x[::24])
        x_tick_labels = [xticklabels[0]] + list(xticklabels[::24])
        ax[i].set_xticks(x_ticks)
        ax[i].set_xticklabels(x_tick_labels, rotation=90)

    for i in [1,2]:
        ax[i].set_ylim([0, max(np.max(q_out), max(q_out_real))+1])
        
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()