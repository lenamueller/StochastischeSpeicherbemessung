import pandas as pd
import numpy as np
from scipy.stats import lognorm, kstest


def pu_weibull(n: int):
    """Return the empirical Pu for n data points (Weibull, 1939)."""
    return [m/(n+1) for m in np.arange(1, n+1, 1)]

def fit_lognv(capacities: list[float], print_parameters: bool = False):
    
    n = len(capacities)
    cap_sort = sorted(capacities)

    # Fit LogNV distribution    
    shape, loc, scale = lognorm.fit(cap_sort)
    
    # Calculate Pu    
    pu_emp = pu_weibull(n=n)
    pu_theo = lognorm.cdf(x=cap_sort, s=shape, loc=loc, scale=scale)

    # Calculate quantiles
    q_emp = cap_sort
    q_theo = lognorm.ppf(pu_emp, shape, loc=loc, scale=scale)
    
    # Calculate fixed quantiles
    q_fixed = np.arange(0,1.01,0.01)
    q_theo_fixed = lognorm.ppf(q_fixed, shape, loc=loc, scale=scale)
    
    if print_parameters:
        print(f"LogNV Parameter: Shape = {shape}, Loc = {loc}, Scale = {scale}")
        print(f"LogNV Mean: {lognorm.mean(shape, loc, scale)}, LogNV Std: {lognorm.std(shape, loc, scale)}")        
        
    result = pd.DataFrame()
    result["Rangzahl [-]"] = np.arange(1, n+1, 1)
    result["Kapazität [hm³]"] = cap_sort
    result["empirische Pu [-]"] = pu_emp
    result["theoretische Pu [-]"] = pu_theo
    result["empirische Quantile [hm³]"] = q_emp
    result["theoretische Quantile [hm³]"] = q_theo
    
    result_2 = pd.DataFrame()
    result_2["Pu [-]"] = [round(i, 2) for i in q_fixed]
    result_2["Kapazität [hm³]"] = q_theo_fixed
    
    return result, result_2

def fit_lognv_kstest(pu_emp: list[float], pu_theo: list[float]):
    return kstest(pu_emp, pu_theo)

def fit_lognv_rqq(q_emp: list[float], q_theo: list[float]) -> float:
    return np.corrcoef(q_emp, q_theo)[0][1]
