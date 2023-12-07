import pandas as pd
import numpy as np
from scipy.stats import skew
import time

from config import pegelname, N_TIMESERIES, N_YEARS, MONTH_HYD_YEAR, MONTH, MONTH_80A
from utils.plotting import plot_monthly_fitting, plot_thomasfierung_eval
from utils.statistics import hyd_years
from utils.data_structures import _monthly_vals


def index_month_before(i: int) -> int:
    """Return the index of the month before."""
    return 12 if i==1 else i-1

def parameter_xp(df: pd.DataFrame, i:int) -> float:
    """Return the parameter x_p for month i."""
    # Extract values of month i
    arr = _monthly_vals(df, i)
    # Calculate monthly mean
    xp = np.mean(arr)
    return xp

def parameter_sp(df: pd.DataFrame, i:int) -> float:
    """Return the parameter s_p for month i."""
    # Extract values of month i
    arr = _monthly_vals(df, i)
    # Calculate monthly mean    
    mean = np.mean(arr)
    # Calculate monthly (unbiased) standard deviation
    sp = np.sqrt(np.sum([(i-mean)**2 for i in arr]) / (len(arr)-1))
    return sp

def parameter_rp(df: pd.DataFrame, i:int) -> float:
    """Return the parameter r_p for month i."""
    
    # Number of hydrological years
    m = len(hyd_years(df))
    
    # Model parameters of month i and i-1
    xp = parameter_xp(df, i)
    xp_before = parameter_xp(df, index_month_before(i))
    sp_i = parameter_sp(df, i)
    sp_i_before = parameter_sp(df, index_month_before(i))

    # Discharge values of month i and i-1
    x_current_month = _monthly_vals(df, i)
    x_month_before = _monthly_vals(df, index_month_before(i))
    
    # November: Use last October value (10/1999) as value of 
    # month before, because there is no previous October
    # value if time series starts in November 1959.
    if i == 11:
        last_val = x_month_before[-1]
        x_month_before = np.insert(x_month_before[:-1], 0, last_val)
        
    # Calculate correlation coefficient
    sum_term = sum([x*y for x,y in zip(x_current_month, x_month_before)])
    rp = (sum_term - m*xp*xp_before) / (sp_i*sp_i_before*(m-1))

    # Check if correlation coefficient is in [-1,1]    
    if -1 <= rp <= 1:
        return rp
    else:
        raise ValueError("r_p is not in [-1,1] for month", i)

def gen_timeseries(df: pd.DataFrame, n_years: int = 1) -> float:
    """
    Returns a time series with monthly discharge values starting
    in November ending in October, generated with the Thomas-Fiering 
    model.
    """

    new_time_series = []
    
    # Iterate over years    
    for _ in range(n_years):
        # Iterate over months
        for i in [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if i == 11:
                # November: use mean of October as value of month before
                value_month_before = parameter_xp(df, 10)
            else:
                # other months: use last value of new time series as value 
                # of month before
                value_month_before = new_time_series[-1]

            # Model parameters of month i and i-1
            xp = parameter_xp(df, i)
            sp = parameter_sp(df, i)
            rp = parameter_rp(df, i)
            xp_before = parameter_xp(df, index_month_before(i))
            sp_before = parameter_sp(df, index_month_before(i))
            rp_before = parameter_rp(df, index_month_before(i))
            
            # Random value from normal distribution with mean 0 and std 1
            ti = np.random.normal(loc=0, scale=1, size=1)[0]
            
            # Unbiased sample skewness
            csi = skew(_monthly_vals(df, i), bias=False)
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

def thomasfiering(raw_data: pd.DataFrame):
    print("\n--------------------------------------")
    print("\nZeitreihengenerierung\n")
    
    start_time = time.time()
    
    # calculate parameters
    tf_pars = pd.DataFrame(
        data={
            "Mittelwert": [parameter_xp(raw_data, i) for i in MONTH_HYD_YEAR],
            "Standardabweichung": [parameter_sp(raw_data, i) for i in MONTH_HYD_YEAR],
            "Korrelationskoeffizient": [parameter_rp(raw_data, i) for i in MONTH_HYD_YEAR]
        },
        index=[MONTH[i] for i in MONTH_HYD_YEAR]
    )

    tf_pars.round(4).to_csv(f"data/{pegelname}_tomasfiering_parameters.csv", index=True)
    tf_pars.round(4).to_latex(f"data/{pegelname}_tomasfiering_parameters.tex", index=True)
    print(f"-> Modellparameter: data/{pegelname}_tomasfiering_parameters.csv")
    
    # generate data
    gen_data = pd.DataFrame()
    # gen_data["Monat"] = MONTH_80A
    
    for i in range(N_TIMESERIES):
        print("\nGeneriere Zeitreihe", i+1, "von", N_TIMESERIES, "...")
        gen_data[f"G{str(i+1).zfill(3)}_m3s"] = gen_timeseries(raw_data, n_years=N_YEARS)
        
    gen_data.to_csv(
        f"data/{pegelname}_thomasfiering_timeseries.csv", index=True)
    gen_data.iloc[:, :11].round(3).to_latex(
        f"data/{pegelname}_thomasfiering_timeseries_first10.tex", index=True)

    print(f"-> generierte Zeitreihen: data/{pegelname}_thomasfiering_timeseries.csv")
    print("Dauer der Generierung:", round(time.time()-start_time, 1), "Sekunden")
    
    plot_thomasfierung_eval(raw_data=raw_data, gen_data=gen_data)
    plot_monthly_fitting(raw_data)
    
    return gen_data