import pandas as pd
import numpy as np
from scipy.stats import lognorm

from config import T
from utils.statistics import hyd_years, standard_deviation_unbiased


def _monthly_vals(df: pd.DataFrame, i: int)-> np.ndarray:
    df = df[df["Monat"].str.startswith(str(i).zfill(2))]
    return df["Durchfluss_m3s"].to_numpy()

def index_month_before(i: int) -> int:
    """Return the index of the month before."""
    if i  == 1:
        return 12
    else:
        return i - 1

def parameter_xp(df: pd.DataFrame, i:int) -> float:
    """Return the parameter x_p for month i."""
    return np.mean(_monthly_vals(df, i))

def parameter_sp(df: pd.DataFrame, i:int) -> float:
    """Return the parameter s_p for month i."""
    arr = _monthly_vals(df, i)
    mean = np.mean(arr)
    sp = np.sqrt(np.sum([(i-mean)**2 for i in arr]) / (len(arr)-1))
    return sp

def parameter_rp(df: pd.DataFrame, i:int) -> float:
    """Return the parameter r_p for month i."""
    
    m = len(hyd_years(df))
    xp = parameter_xp(df, i)
    xp_before = parameter_xp(df, index_month_before(i))
    sp_i = parameter_sp(df, i)
    sp_i_before = parameter_sp(df, index_month_before(i))

    x_current_month = _monthly_vals(df, i)
    x_month_before = _monthly_vals(df, index_month_before(i))
    
    if i == 11:
        # For the first November the value from the last (not previous)
        # October is used!
        last_val = x_month_before[-1]
        x_month_before = np.insert(x_month_before[:-1], 0, last_val)
        
    sum_term = sum([x*y for x,y in zip(x_current_month, x_month_before)])
    rp = (sum_term - m*xp*xp_before) / (sp_i*sp_i_before*(m-1))
    
    if -1 <= rp <= 1:
        return rp
    else:
        raise ValueError("r_p is not in [-1,1] for month", i)

def fit_lognorm(df: pd.DataFrame, i: int) -> tuple[float, float, float]:
    x = _monthly_vals(df, i)
    shape, loc, scale = lognorm.fit(x)
    return shape, loc, scale

