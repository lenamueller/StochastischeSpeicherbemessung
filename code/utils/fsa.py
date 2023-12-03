import pandas as pd
import numpy as np

from utils.plotting import plot_monthly_discharge, plot_fsa, \
    plot_storage
from config import pegelname, ALPHA, ABGABEN, SEC_PER_MONTH, \
    N_TIMESERIES, MONTH_HYD_YEAR_TXT


def monthly_discharge(arr: np.ndarray) -> dict:
    """Calculate monthly discharge for a given time series."""

    # Transform discharge to volume in hm³
    arr_hm3 = np.array([i* SEC_PER_MONTH/1000000 for i in arr])

    # Calculate yearly sums
    arr_hm3 = np.reshape(arr_hm3, (40, 12))
    yearly_sums = np.sum(arr_hm3, axis=1)

    # Calculate mean yearly sum    
    mean_discharge = np.mean(yearly_sums, axis=0)
    
    # Calculate monthly discharge
    return {
        11:     mean_discharge * ABGABEN[11]/100 * ALPHA,
        12:     mean_discharge * ABGABEN[12]/100 * ALPHA,
        1:      mean_discharge * ABGABEN[1]/100 * ALPHA,
        2:      mean_discharge * ABGABEN[2]/100 * ALPHA,
        3:      mean_discharge * ABGABEN[3]/100 * ALPHA,
        4:      mean_discharge * ABGABEN[4]/100 * ALPHA,
        5:      mean_discharge * ABGABEN[5]/100 * ALPHA,
        6:      mean_discharge * ABGABEN[6]/100 * ALPHA,
        7:      mean_discharge * ABGABEN[7]/100 * ALPHA,
        8:      mean_discharge * ABGABEN[8]/100 * ALPHA,
        9:      mean_discharge * ABGABEN[9]/100 * ALPHA,
        10:     mean_discharge * ABGABEN[10]/100 * ALPHA
    }

def storage_sim(
        q_in: np.ndarray, 
        q_out: np.ndarray, 
        initial_storage: float = 0,
        max_cap: float = np.inf
        ) -> tuple[list[float], list[float], list[float], list[float]]:
    
    storage = []
    deficit = []
    overflow = []    
    q_out_real = [] # Ist-Abgabe

    # Initial storage    
    current_storage = initial_storage
    
    for i in range(len(q_in)):
        
        # Add netto inflow to current storage
        current_storage += q_in[i] - q_out[i]
        
        # Empty storage
        if current_storage < 0:
            storage.append(0)
            deficit.append(current_storage)
            overflow.append(0)
            q_out_real.append(q_out[i]-current_storage)
            
            current_storage = 0

        # Full storage
        elif current_storage > max_cap:
            
            storage.append(max_cap)
            deficit.append(0)
            overflow.append(current_storage-max_cap)
            q_out_real.append(q_out[i]+current_storage-max_cap)
            
            current_storage = max_cap

        # Normal storage
        else:
            if current_storage < 0:
                raise ValueError("Negative storage!")
            else:
                storage.append(current_storage)
                deficit.append(0)
                overflow.append(0)
                q_out_real.append(q_out[i])
    
    return storage, deficit, overflow, q_out_real

def calc_maxima(storage: np.ndarray) -> tuple[list[float], list[float]]:
    """A maximum is a value greater than its predecessor and 
    its successor and all following maxima must be greater 
    than the previous one."""
    
    max_vals = []
    max_indices = []
    for i in range(1, len(storage)-1):
        if storage[i-1] < storage[i] and storage[i] > storage[i+1]:
            if len(max_vals) == 0:
                max_vals.append(storage[i])
                max_indices.append(i)
            else:
                if storage[i] >= max_vals[-1]:
                    max_vals.append(storage[i])
                    max_indices.append(i)
    
    return max_vals, max_indices

def calc_minima(
        storage: np.ndarray, 
        max_indices: np.ndarray
        ) -> tuple[list[float], list[float]]:
    """A minima is the smallest value between two maxima which
    locations are given as max_indices."""
    
    min_vals = []
    min_indices = []
    for i in range(len(max_indices)-1):
        min_vals.append(min(storage[max_indices[i]:max_indices[i+1]]))
        min_indices.append(np.argmin(storage[max_indices[i]:max_indices[i+1]]) + max_indices[i])
    
    return min_vals, min_indices

def calc_capacity(storage: np.ndarray) -> tuple[float, int]:
    
    # Calculate maxima and minima of SDL
    max_vals, max_indices = calc_maxima(storage)
    min_vals, min_indices = calc_minima(storage, max_indices)

    # Remove last maximum because no minimum can follow.
    max_vals = max_vals[:-1]
    max_indices = max_indices[:-1]
    
    # calculate differences between maxima and minima
    assert len(max_vals) == len(min_vals)
    diff = [i-j for i, j in zip(max_vals, min_vals)]

    # get maximum difference and its location
    if diff != []:
        cap = max(diff)
        cap_min = min_vals[diff.index(cap)]
        cap_max = max_vals[diff.index(cap)]
        cap_min_index = min_indices[diff.index(cap)]
        cap_max_index = max_indices[diff.index(cap)]
        return cap, cap_min_index, cap_min, cap_max_index, cap_max
    else: 
        return 0, 0, 0, 0, 0

def fsa(raw_data: pd.DataFrame, gen_data: pd.DataFrame):
    
    print("\n--------------------------------------")
    print("\nBerechnung monatlicher Soll-Abgaben\n")
    
    monthly_dis = {}
    monthly_dis["original"] = monthly_discharge(raw_data["Durchfluss_m3s"].to_numpy())
    for i in range(N_TIMESERIES):
        monthly_dis[f"G{str(i+1).zfill(3)}"] = monthly_discharge(
            gen_data[f"G{str(i+1).zfill(3)}_m3s"].to_numpy())

    monthly_dis = pd.DataFrame.from_dict(monthly_dis)
    monthly_dis.index = MONTH_HYD_YEAR_TXT
    monthly_dis = monthly_dis.transpose()
    monthly_dis.round(3).to_csv(f"data/{pegelname}_monthly_discharge.csv", index=True)
    monthly_dis.round(3).to_latex(f"data/{pegelname}_monthly_discharge.tex", index=True)
    print(f"-> data/{pegelname}_monthly_discharge.csv")
    
    plot_monthly_discharge(monthly_dis)
    
    print("\n--------------------------------------")
    print("\nBerechnung Speicherkapazität\n")
    
    capacities = {}
    
    # -----------------------------------------
    #   convert inflow from m³/s to hm³
    # -----------------------------------------
    
    raw_data["Durchfluss_hm3"] = raw_data["Durchfluss_m3s"] * SEC_PER_MONTH/1000000
    print(raw_data)
    print(gen_data)
    # gen_data = gen_data.transpose()
    # print(gen_data)
    
    for i in range(N_TIMESERIES):
        gen_data[f"G{str(i+1).zfill(3)}_hm3"] = \
            gen_data[f"G{str(i+1).zfill(3)}_m3s"] * SEC_PER_MONTH/1000000
    print(gen_data)
    
    # -----------------------------------------
    #               original data
    # -----------------------------------------
    
    q_in = raw_data["Durchfluss_hm3"].to_numpy()
    q_out = np.tile(monthly_dis.loc["original", :].to_numpy(), 40)
    
    storage, _, _, _ = storage_sim(q_in, q_out, initial_storage=0, max_cap=np.inf)
    
    cap, cap_min_index, cap_min, _, cap_max = calc_capacity(storage)
    capacities["original"] = cap
    
    # Plot FSA for original data
    max_vals, max_indices = calc_maxima(storage=storage)
    min_vals, min_indices = calc_minima(storage=storage, max_indices=max_indices)
    plot_fsa(storage, max_vals=max_vals, max_indices=max_indices, 
             min_vals=min_vals, min_indices=min_indices, cap=cap, 
             cap_min_index=cap_min_index, cap_min=cap_min, cap_max=cap_max)

    # Plot storage simulation for unlimited reservoir
    storage, deficit, overflow, q_out_real = storage_sim(
        q_in, q_out, initial_storage=0, max_cap=np.inf)
    plot_storage(q_in, q_out, q_out_real, storage, deficit, overflow, 
                fn_ending="unlimited")

    # Plot storage simulation for limited reservoir
    storage, deficit, overflow, q_out_real = storage_sim(
        q_in, q_out, initial_storage=0, max_cap=cap)
    plot_storage(q_in, q_out, q_out_real, storage, deficit, overflow, 
                fn_ending=str(round(cap, 3)))
    
    # -----------------------------------------
    #            generated data
    # -----------------------------------------
    
    for i in range(N_TIMESERIES):
        q_in = gen_data[f"G{str(i+1).zfill(3)}_hm3"]
        q_out = np.tile(monthly_dis.loc[f"G{str(i+1).zfill(3)}", :].to_numpy(), 40)
        storage, _, _, _ = storage_sim(q_in, q_out, initial_storage=0, max_cap=np.inf)
        cap, _, _, _, _ = calc_capacity(storage)
        capacities[f"G{str(i+1).zfill(3)}"] = cap

    df_capacities = pd.DataFrame()
    df_capacities["Zeitreihe"] = capacities.keys()
    df_capacities["Kapazität"] = capacities.values()

    df_capacities.to_csv(f"data/{pegelname}_capacities.csv", index=False)
    df_capacities.to_latex(f"data/{pegelname}_capacities.tex", index=False)
    
    print(df_capacities)