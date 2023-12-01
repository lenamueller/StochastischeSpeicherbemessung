import pandas as pd
import numpy as np

from utils.statistics import binned_stats
from config import ALPHA, ABGABEN, SEC_PER_MONTH


def monthly_discharge(df: pd.DataFrame) -> dict:
    """Calculate monthly discharge for a given time series."""
    
    df["Durchfluss_hm3"] = df["Durchfluss_m3s"] * SEC_PER_MONTH / 1000000
    yearly_sums = binned_stats(df, var="Durchfluss_hm3", bin="yearly", func=np.sum)
    mean_discharge = np.mean(yearly_sums)
    
    return {
        11: mean_discharge * ABGABEN[11]/100 * ALPHA,
        12: mean_discharge * ABGABEN[12]/100 * ALPHA,
        1: mean_discharge * ABGABEN[1]/100 * ALPHA,
        2: mean_discharge * ABGABEN[2]/100 * ALPHA,
        3: mean_discharge * ABGABEN[3]/100 * ALPHA,
        4: mean_discharge * ABGABEN[4]/100 * ALPHA,
        5: mean_discharge * ABGABEN[5]/100 * ALPHA,
        6: mean_discharge * ABGABEN[6]/100 * ALPHA,
        7: mean_discharge * ABGABEN[7]/100 * ALPHA,
        8: mean_discharge * ABGABEN[8]/100 * ALPHA,
        9: mean_discharge * ABGABEN[9]/100 * ALPHA,
        10: mean_discharge * ABGABEN[10]/100 * ALPHA
    }

def calc_sdl(
        q_in: np.ndarray, 
        q_out: np.ndarray, 
        initial_storage: float = 0,
        max_cap: float = np.inf
        ):
    
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