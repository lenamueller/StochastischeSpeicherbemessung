import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from setup import tu_mediumblue

line_kwargs = {
    "color": tu_mediumblue,
    "linewidth": 0.8
}

def visualize_discharge(
        monthly_data: pd.DataFrame, 
        yearly_data: pd.DataFrame,
        fn: str
        ):

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # plot monthly discharge
    axs[0].plot(monthly_data["Monat"], monthly_data["Durchfluss"], **line_kwargs)
    axs[0].set_xlabel("Monat")
    axs[0].set_ylabel("Durchfluss [m³/s]")
    axs[0].set_xticks(monthly_data["Monat"][::12])
    axs[0].set_xticklabels(monthly_data["Monat"][::12], rotation=90)
    axs[0].set_yticks(np.arange(0, max(monthly_data["Durchfluss"]), 1), minor=False)
    axs[0].set_yticks(np.arange(0, max(monthly_data["Durchfluss"]), 0.25), minor=True)
    
    # plot yearly discharge
    axs[1].plot(yearly_data["Jahr"], yearly_data["Durchfluss"], **line_kwargs)
    axs[1].set_xlabel("Jahr")
    axs[1].set_ylabel("Durchfluss [m³/s]")
    axs[1].set_xticks(yearly_data["Jahr"][::2])
    axs[1].set_xticklabels(yearly_data["Jahr"][::2], rotation=90)
    axs[1].set_yticks(np.arange(0, max(yearly_data["Durchfluss"]), 0.5), minor=False)
    axs[1].set_yticks(np.arange(0, max(yearly_data["Durchfluss"]), 0.1), minor=True)

    # settings
    for i in [0, 1]:
        axs[i].grid(which="major", axis="y", color="grey", alpha=0.75)
        axs[i].grid(which="minor", axis="y", color="grey", alpha=0.25)
        axs[i].set_ylim(bottom=0)
    
    # save figure
    fig.tight_layout()
    plt.savefig("images/" + fn)
    return None
