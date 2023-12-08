import pandas as pd
import numpy as np

from settings import SEC_PER_MONTH, ABGABEN, SPEICHERAUSGLEICHSGRAD, PEGEL


def convert_m3s_hm3(x):
    return x * SEC_PER_MONTH/1000000
    
def soll_abgabe(timeseries: list[float]) -> dict:
    """Calculate monthly discharge for a given time series."""

    # Transform discharge to volume in hm³
    arr_hm3 = np.array([convert_m3s_hm3(i) for i in timeseries])

    # Calculate yearly sums
    arr_hm3 = np.reshape(arr_hm3, (int(len(arr_hm3)/12), 12))
    yearly_sums = np.sum(arr_hm3, axis=1)

    # Calculate mean yearly sum    
    mean_discharge = np.mean(yearly_sums, axis=0)
    
    # Calculate monthly discharge
    return {
        11:     mean_discharge * ABGABEN[11]/100 * SPEICHERAUSGLEICHSGRAD,
        12:     mean_discharge * ABGABEN[12]/100 * SPEICHERAUSGLEICHSGRAD,
        1:      mean_discharge * ABGABEN[1]/100 * SPEICHERAUSGLEICHSGRAD,
        2:      mean_discharge * ABGABEN[2]/100 * SPEICHERAUSGLEICHSGRAD,
        3:      mean_discharge * ABGABEN[3]/100 * SPEICHERAUSGLEICHSGRAD,
        4:      mean_discharge * ABGABEN[4]/100 * SPEICHERAUSGLEICHSGRAD,
        5:      mean_discharge * ABGABEN[5]/100 * SPEICHERAUSGLEICHSGRAD,
        6:      mean_discharge * ABGABEN[6]/100 * SPEICHERAUSGLEICHSGRAD,
        7:      mean_discharge * ABGABEN[7]/100 * SPEICHERAUSGLEICHSGRAD,
        8:      mean_discharge * ABGABEN[8]/100 * SPEICHERAUSGLEICHSGRAD,
        9:      mean_discharge * ABGABEN[9]/100 * SPEICHERAUSGLEICHSGRAD,
        10:     mean_discharge * ABGABEN[10]/100 * SPEICHERAUSGLEICHSGRAD
    }

def calc_storage_simulation(
        q_in: np.ndarray, 
        q_out_soll: np.ndarray, 
        initial_storage: float = 0,
        max_cap: float = np.inf,
        q_in_convert: bool = True,
        q_out_soll_convert: bool = False,
        ) -> tuple[list[float], list[float], list[float], list[float]]:
    
    # convert inflow from m³/s to hm³ if necessary
    if q_in_convert:
        q_in = [convert_m3s_hm3(i) for i in q_in]
    if q_out_soll_convert:
        q_out_soll = [convert_m3s_hm3(i) for i in q_out_soll]
    
    storage = []
    deficit = []
    overflow = []    
    q_out_ist = [] # Ist-Abgabe

    # Initial storage    
    current_storage = initial_storage
    
    # Start simulation
    for i in range(len(q_in)):
        
        # Add netto inflow to current storage
        current_storage += q_in[i] - q_out_soll[i]
        
        # Empty storage
        if current_storage < 0:
            storage.append(0)
            deficit.append(current_storage)
            overflow.append(0)
            q_out_ist.append(q_out_soll[i]-current_storage)
            
            current_storage = 0

        # Full storage
        elif current_storage > max_cap:
            
            storage.append(max_cap)
            deficit.append(0)
            overflow.append(current_storage-max_cap)
            q_out_ist.append(q_out_soll[i]+current_storage-max_cap)
            
            current_storage = max_cap

        # Normal storage
        else:
            if current_storage < 0:
                raise ValueError("Negative storage!")
            else:
                storage.append(current_storage)
                deficit.append(0)
                overflow.append(0)
                q_out_ist.append(q_out_soll[i])
    
    return storage, deficit, overflow, q_out_ist

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




# # -----------------------------------------
# # plot FSA for original data
# # -----------------------------------------

# max_vals, max_indices = calc_maxima(storage=storage)
# min_vals, min_indices = calc_minima(storage=storage, max_indices=max_indices)
# plot_fsa(storage, max_vals=max_vals, max_indices=max_indices, 
#             min_vals=min_vals, min_indices=min_indices, cap=cap, 
#             cap_min_index=cap_min_index, cap_min=cap_min, cap_max=cap_max)
