import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.statistics import binned_stats
from config import ALPHA, ABGABEN, SEC_PER_MONTH


def monthly_discharge(df: pd.DataFrame) -> dict:
    """Calculate monthly discharge for a given time series."""
    
    df["Durchfluss_hm3"] = df["Durchfluss_m3s"] * SEC_PER_MONTH / 1000000
    yearly_sums = binned_stats(df, var="Durchfluss_hm3", bin="yearly", func=np.sum)
    mean_discharge = np.mean(yearly_sums)
    
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

def calc_storage(
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
    cap = max(diff)
    cap_min = min_vals[diff.index(cap)]
    cap_max = max_vals[diff.index(cap)]
    cap_min_index = min_indices[diff.index(cap)]
    cap_max_index = max_indices[diff.index(cap)]
    
    return cap, cap_min_index, cap_min, cap_max_index, cap_max