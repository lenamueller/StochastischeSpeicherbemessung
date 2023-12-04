import pandas as pd
import numpy as np
from scipy.stats import lognorm, kstest

from config import pegelname
from utils.plotting import plot_capacity, qq_plot


def pu_weibull(n: int):
    """Return the empirical Pu for n data points (Weibull, 1939)."""
    return [m/(n+1) for m in range(n)]

def fit_capacity():
    print("\n--------------------------------------")
    print("\nKapazität für 90 % Zuverlässigkeit\n")
    
    # read capacities
    cap = pd.read_csv(f"data/{pegelname}_capacities_100.csv")
    cap = cap[cap["Zeitreihe"] != "original"]
    cap_sort = sorted(cap["Kapazität"])
    
    pu_emp = pu_weibull(n=len(cap))
    shape, loc, scale = lognorm.fit(cap_sort)
    print(f"LogNV Parameter: Shape = {shape}, Loc = {loc}, Scale = {scale}")
    print(f"Stats: {lognorm.stats(shape, loc, scale)}")
    
    pu_theo = lognorm.cdf(x=cap_sort, s=shape, loc=loc, scale=scale)

    # quantiles from data
    q_emp = cap_sort
    q_theo = lognorm.ppf(pu_emp, shape, loc=loc, scale=scale)
    
    pd.DataFrame(data={"empQuantile": q_emp, "theoQuantile": q_theo}).round(3).to_csv(
        f"data/{pegelname}_quantile.csv", index=False)
    pd.DataFrame(data={"empQuantile": q_emp, "theoQuantile": q_theo}).round(3).to_latex(
        f"data/{pegelname}_quantile.tex", index=False)
    
    # fixed quantiles
    q = np.arange(0,1,0.05)
    q_theo_fixed = lognorm.ppf(q, shape, loc=loc, scale=scale)

    pd.DataFrame(data={"Quantile":q, "theoQuantile":q_theo_fixed}).round(3).to_csv(
        f"data/{pegelname}_quantile_fixed.csv", index=False)
    pd.DataFrame(data={"Quantile":q, "theoQuantile":q_theo_fixed}).round(3).to_latex(
        f"data/{pegelname}_quantile_fixed.tex", index=False)
    
    # 90 % quantile
    cap_90 = lognorm.ppf(0.9, shape, loc=loc, scale=scale)
    print(f"90 % Quantile: {cap_90}")
    
    plot_capacity(capacities_sort=cap_sort, pu_emp=pu_emp, pu_theo=pu_theo, 
                  cap_90=cap_90)
    qq_plot(emp=q_emp, theo=q_theo)
    
    # KS test
    print(kstest(pu_emp, pu_theo))
    
    
    
    
    

