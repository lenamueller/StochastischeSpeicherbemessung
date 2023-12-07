import pandas as pd
import numpy as np
from scipy.stats import lognorm, kstest

from config import pegelname
from utils.plotting import plot_pu, qq_plot, plot_capacities_hist


def pu_weibull(n: int):
    """Return the empirical Pu for n data points (Weibull, 1939)."""
    return [m/(n+1) for m in range(n)]

def fit_capacity():
    print("\n--------------------------------------")
    print("\nKapazität für 90 % Zuverlässigkeit\n")
    
    # -----------------------------------------
    # read capacities
    # -----------------------------------------
    
    cap = pd.read_csv(f"data/{pegelname}_capacities.csv") # always use sample size 100
    cap_hist = cap[cap["Zeitreihe"] == "original"].to_numpy()[0][1:][0]
    cap = cap[cap["Zeitreihe"] != "original"]
    cap_sort = sorted(cap["Kapazität"])
    print("Min. Kapazität: ", cap_sort[0])
    print("Max. Kapazität: ", cap_sort[-1])
    
    # -----------------------------------------
    # emp. and theo. Pu
    # -----------------------------------------
    
    rangzahlen = np.arange(1, len(cap)+1)
    pu_emp = pu_weibull(n=len(cap))

    shape, loc, scale = lognorm.fit(cap_sort)
    print(f"LogNV Parameter: Shape = {shape}, Loc = {loc}, Scale = {scale}")
    print(f"LogNV Mean: {lognorm.mean(shape, loc, scale)}, LogNV Std: {lognorm.std(shape, loc, scale)}")
    pu_theo = lognorm.cdf(x=cap_sort, s=shape, loc=loc, scale=scale)

    pd.DataFrame(data={"Rangzahl [-]": rangzahlen, "Kapazität [hm³]": cap_sort, "empirische Pu [-]": pu_emp, "theoretische Pu  [-]": pu_theo}).round(3).to_csv(
        f"data/{pegelname}_pu.csv", index=False)
    pd.DataFrame(data={"Rangzahl [-]": rangzahlen, "Kapazität [hm³]": cap_sort, "empirische Pu [-]": pu_emp, "theoretische Pu [-]": pu_theo}).round(3).to_latex(
        f"data/{pegelname}_pu.tex", index=False, float_format="%.3f")
        
    # -----------------------------------------
    # emp. and theo. quantiles
    # -----------------------------------------
    
    q_emp = cap_sort
    q_theo = lognorm.ppf(pu_emp, shape, loc=loc, scale=scale)
    
    pd.DataFrame(data={"empirische Quantile [hm³]": q_emp, "theoretische Quantile [hm³]": q_theo}).round(3).to_csv(
        f"data/{pegelname}_quantile.csv", index=True)
    pd.DataFrame(data={"empirsiche Quantile [hm³]": q_emp, "theoretische Quantile [hm³]": q_theo}).round(3).to_latex(
        f"data/{pegelname}_quantile.tex", index=True, float_format="%.3f")
    
    # -----------------------------------------
    # theo. quantiles for fixed Pu
    # -----------------------------------------
    
    q = np.arange(0,1,0.05)
    q_theo_fixed = lognorm.ppf(q, shape, loc=loc, scale=scale)

    pd.DataFrame(data={"Pu [-]":q, "theoretische Quantile [hm³]":q_theo_fixed}).round(3).to_csv(
        f"data/{pegelname}_quantile_fixed.csv", index=False)
    pd.DataFrame(data={"Pu [-]":q, "theoretische Quantile [hm³]":q_theo_fixed}).round(3).to_latex(
        f"data/{pegelname}_quantile_fixed.tex", index=False, float_format="%.3f")
    
    # 90 % quantile
    cap_90 = lognorm.ppf(0.9, shape, loc=loc, scale=scale)
    print(f"90 % Quantile: {cap_90}")
    
    # -----------------------------------------
    # test fitting
    # -----------------------------------------
    
    # KS test
    print(kstest(pu_emp, pu_theo))
    
    # rqq test
    print(f"r_qq = {np.corrcoef(q_emp, q_theo)[0][1]}")
    
    # -----------------------------------------
    # plot
    # -----------------------------------------

    plot_pu(capacities_sort=cap_sort, pu_emp=pu_emp, pu_theo=pu_theo, cap_90=cap_90)
    plot_capacities_hist(cap["Kapazität"], cap_hist)
    qq_plot(emp=q_emp, theo=q_theo)