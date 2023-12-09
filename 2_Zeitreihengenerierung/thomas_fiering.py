import pandas as pd
import numpy as np
from scipy.stats import skew

from utils.read import monthly_vals
from settings import N_GEN_YEARS


def index_month_before(i: int) -> int:
    """Return the index of the month before."""
    return 12 if i==1 else i-1

def parameter_xp(df: pd.DataFrame, i:int) -> float:
    """Return the parameter x_p for month i."""
    # Extract values of month i
    arr = monthly_vals(df, i)
    # Calculate monthly mean
    xp = np.mean(arr)
    return xp

def parameter_sp(df: pd.DataFrame, i:int) -> float:
    """Return the parameter s_p for month i."""
    # Extract values of month i
    arr = monthly_vals(df, i)
    # Calculate monthly mean    
    mean = np.mean(arr)
    # Calculate monthly (unbiased) standard deviation
    sp = np.sqrt(np.sum([(i-mean)**2 for i in arr]) / (len(arr)-1))
    return sp

def parameter_rp(df: pd.DataFrame, i:int) -> float:
    """Return the parameter r_p for month i."""
    
    # Number of hydrological years
    m = len(monthly_vals(df, 1))
    
    # Model parameters of month i and i-1
    xp = parameter_xp(df, i)
    xp_before = parameter_xp(df, index_month_before(i))
    sp_i = parameter_sp(df, i)
    sp_i_before = parameter_sp(df, index_month_before(i))

    # Discharge values of month i and i-1
    x_current_month = monthly_vals(df, i)
    x_month_before = monthly_vals(df, index_month_before(i))
    
    if i == 11:
        # Reorder values
        last_val = x_month_before[-1]
        x_month_before = np.insert(x_month_before[:-1], 0, last_val)
        
        # Ignore first year because the first November value has 
        # no previous October value. Otherwise, the correlation
        # would be calculated between 11/1959 and 10/19999.
        x_current_month = x_current_month[1:]
        x_month_before = x_month_before[1:]
        
        
    # Calculate correlation coefficient
    sum_term = sum([x*y for x,y in zip(x_current_month, x_month_before)])
    rp = (sum_term - m*xp*xp_before) / (sp_i*sp_i_before*(m-1))

    # Check if correlation coefficient is in [-1,1]    
    if -1 <= rp <= 1:
        return rp
    else:
        raise ValueError("r_p is not in [-1,1] for month", i)

def gen_timeseries(df: pd.DataFrame) -> float:
    """
    Returns a time series with monthly discharge values starting
    in November ending in October, generated with the Thomas-Fiering 
    model.
    """

    new_time_series = []
    
    for _ in range(N_GEN_YEARS):

        for month in [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if month == 11:
                # November: use mean of October as value of month before
                value_month_before = parameter_xp(df, 10)
            else:
                # other months: use last value of new time series as value 
                # of month before
                value_month_before = new_time_series[-1]

            # Model parameters of month i and i-1
            xp = parameter_xp(df, month)
            sp = parameter_sp(df, month)
            rp = parameter_rp(df, month)
            xp_before = parameter_xp(df, index_month_before(month))
            sp_before = parameter_sp(df, index_month_before(month))
            rp_before = parameter_rp(df, index_month_before(month))
            
            # Random value from normal distribution with mean 0 and std 1
            ti = np.random.normal(loc=0, scale=1, size=1)[0]
            
            # Unbiased sample skewness
            csi = skew(monthly_vals(df, month), bias=False)
            cti = (csi - csi*(rp_before**3) - 1) / (((1 - rp**2))**(3/2))
            
            # Random value from gamma distribution
            tg = 2/cti * (1 + (cti*ti)/6 - (cti**2)/36)**3 - 2/cti

            # Thomas-Fiering model equation
            term1 = xp
            term2 = (rp * sp/sp_before) * (value_month_before - xp_before)
            term3 = tg*sp*np.sqrt(1-rp**2)
            
            # New value for month i
            x_new = term1 + term2 + term3
            new_time_series.append(x_new)

    return new_time_series
