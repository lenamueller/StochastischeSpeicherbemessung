import numpy as np
import pandas as pd


def rel_yearly(
        deficit: np.ndarray,
        overflow: np.ndarray,
        n_years: int
        ):
    
    """Return reliability metrics for yearly values."""
    
    # convert to array with shape 40 rows and 12 columns
    deficit = deficit.reshape((40, 12))
    overflow = overflow.reshape((40, 12))
    
    # number of months with deficit and overflow
    n_def = 0
    n_ovf = 0
    for i in range(len(deficit)):
        if np.sum(deficit[i]) > 0:
            n_def += 1
        if np.sum(overflow[i]) > 0:
            n_ovf += 1
    
    # relibaility metrics
    rel_deficit = round(1 - n_def / n_years, 3)
    rel_overflow = round(1 - n_ovf / n_years, 3)
    
    return rel_deficit, rel_overflow

def rel_monthly(
    deficit: np.ndarray,
    overflow: np.ndarray,
    n_years: int
    ):
    """Return reliability metrics for monthly values."""
    
    # number of months with deficit and overflow
    n_def = 0
    n_ovf = 0
    for i in range(len(deficit)):
        if deficit[i] > 0:
            n_def += 1
        if overflow[i] > 0:
            n_ovf += 1
        
    # relibaility metrics
    rel_deficit = round(1 - n_def / (n_years*12), 3)
    rel_overflow = round(1 - n_ovf / (n_years*12), 3)
    
    return rel_deficit, rel_overflow

def rel_amount(
    deficit: np.ndarray,
    overflow: np.ndarray,
    soll_abgabe: np.ndarray
        ): 
    """Return the amout of deficit and overflow from Soll-Abgabe."""
    
    sum_deficit = np.sum(deficit)
    sum_overflow = np.sum(overflow)
    sum_soll_abgabe = np.sum(soll_abgabe)

    r_def = sum_deficit / sum_soll_abgabe
    r_ovf = sum_overflow / sum_soll_abgabe

    return r_def, r_ovf

def reliability(fn: list[str]):
    
    print("\n--------------------------------------")
    print(f"\nZuverlässigkeitsprüfung für Zeitreihen {fn}\n")
    
    for fn_i in fn:

        # -----------------------------------------
        # read data
        # -----------------------------------------
        
        df = pd.read_csv(fn_i)
        print(df)
        deficit = df["deficit"].to_numpy()
        overflow = df["overflow"].to_numpy()
        soll_abgabe = df["Soll-Abgabe [hm³]"].to_numpy()
        
        # -----------------------------------------
        # calculate reliability
        # -----------------------------------------
        
        print("Anzahl Monate", rel_monthly(deficit=deficit, overflow=overflow, n_years=80))
        print("Anzahl Jahre", rel_yearly(deficit=deficit, overflow=overflow, n_years=80))
        print("Summe Sollabgabe", np.sum(soll_abgabe), "hm³")
        print("Menge", rel_amount(deficit=deficit, overflow=overflow, soll_abgabe=soll_abgabe))
        
        