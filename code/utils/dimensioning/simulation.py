import pandas as pd
import numpy as np

from utils.data_structures import read_gen_data, read_data
from utils.dimensioning.fsa import calc_storage_simulation
from utils.plotting import plot_storage_simulation, plot_deficit_overflow

from config import pegelname, SEC_PER_MONTH


def get_q_out(var: str) -> np.ndarray:
    """Return Soll-Abgabe for a given time series"""
    df_out = pd.read_csv("data/Klingenthal_monthly_discharge.csv", index_col=0)
    return np.tile(df_out.loc[var].to_numpy(), reps=40)


def run_simulation(
        var: str,
        cap: float,
        initial_storage: float,
    ):
    
    """ 
    raw data:           var = "original"
    generated data:     var = "G_xxx"
    """
    
    print("\n--------------------------------------")
    print(f"\nSeichersimulation für Zeitreihe {var} \n\tmit Kapazität {cap} und \n\tAnfangsfüllung {initial_storage}\n")
    
    # -----------------------------------------
    # read inflow and outflow
    # -----------------------------------------
    
    if var == "original":
        raw_data = read_data("data/Klingenthal_raw.txt")
        months = raw_data["Monat"].to_numpy()
        q_in = raw_data["Durchfluss_m3s"].to_numpy() * SEC_PER_MONTH/1000000
        q_out = get_q_out(var)

    else:
        gen_data = read_gen_data()
        months = gen_data.index.to_numpy()
        q_in = gen_data[f"{var}_m3s"].to_numpy() * SEC_PER_MONTH/1000000
        q_in[q_in < 0] = 0
        q_out = get_q_out(var)
    
    # -----------------------------------------
    # simulate storage
    # -----------------------------------------
    
    storage, deficit, overflow, q_out_real = calc_storage_simulation(
        q_in, q_out, initial_storage=initial_storage, max_cap=cap)
    
    # -----------------------------------------
    # plot
    # -----------------------------------------

    plot_storage_simulation(q_in, q_out, q_out_real, storage, deficit, overflow, 
                var=var, cap=cap, initial_storage=initial_storage, 
                xticklabels=months)
    
    plot_deficit_overflow(deficit, overflow, months=months,
                          var=var, cap=cap, initial_storage=initial_storage)

    # -----------------------------------------
    # save data
    # -----------------------------------------
    if cap == np.inf:
        cap_str = "inf"
    else: 
        cap_str = str(round(cap, 3))
            
    pd.DataFrame(data={
        "Monat": months,
        "Zufluss [hm³]": q_in,
        "Soll-Abgabe [hm³]": q_out,
        "Ist-Abgabe [hm³]": q_out_real,
        "Speicherinhalt [hm³]": storage, 
        "Defizit [hm³]": deficit, 
        "Überschuss [hm³]": overflow
        }
                 ).to_csv(
        f"data/{pegelname}_storagesim_{var}_{str(round(initial_storage, 3))}_{cap_str}.csv", 
        index=False, float_format="%.3f")