import pandas as pd
import numpy as np
from scipy.stats import skew

from utils.statistics import hyd_years
from utils.data_structures import _monthly_vals


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

def thomasfiering(df: pd.DataFrame) -> float:
    """
    Returns a time series with monthly discharge values 
    generated with the Thomas Fiering model.
    """

    new_time_series = []
    for i in [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if i == 11:
            value_month_before = parameter_xp(df, 10)
        else:
            value_month_before = new_time_series[-1]

        # model parameters
        xp = parameter_xp(df, i)
        sp = parameter_sp(df, i)
        rp = parameter_rp(df, i)
        xp_before = parameter_xp(df, index_month_before(i))
        sp_before = parameter_sp(df, index_month_before(i))
        rp_before = parameter_rp(df, index_month_before(i))
        
        x_new = -1
        while x_new <= 0.1:
            ti = np.random.normal(loc=0, scale=1, size=1)[0] # random value from normal distr.
            csi = skew(_monthly_vals(df, i), bias=False) # sample skewness
            cti = (csi - rp_before**3*csi - 1) / ((1 - rp**2))**(3/2)
            tg = 2/cti * (1 + (cti*ti)/6 - (cti**2)/36)**3 - 2/cti # random value from gamma distr.

            term1 = xp # mean term
            term2 = (rp * sp/sp_before) * (value_month_before - xp_before) # middle term
            term3 = tg*sp*np.sqrt(1-rp**2) # random term
            x_new = term1 + term2 + term3
        
        if x_new <= 0:
            raise ValueError("Negative value generated!")
        
        new_time_series.append(x_new)

    return new_time_series