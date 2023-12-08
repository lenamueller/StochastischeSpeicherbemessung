from types import FunctionType
import pandas as pd
import numpy as np
import pyhomogeneity as hg

from utils.plotting import plot_dsk, plot_breakpoint


def double_sum(test_gauge: list[float], ref_gauge: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Returns a list of double sums."""
    sum_test = np.cumsum(test_gauge)
    sum_ref = np.cumsum(ref_gauge)
    return sum_test, sum_ref

def pettitt_test(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> FunctionType:
    """Test for inhomogeneity using the Pettitt test."""
    return hg.pettitt_test(x=df[var], alpha=0.05)

def homogenity_check(test_pegel: pd.DataFrame, ref_pegel: pd.DataFrame) -> None:
    print("\n--------------------------------------")
    print("\n\tHomogenitätsprüfung\n")
    
    # double sum analysis
    test, ref = double_sum(test_pegel["Durchfluss_m3s"], ref_pegel["Durchfluss_m3s"])
    plot_dsk(test, ref)
    
    # break point analysis
    res = pettitt_test(test_pegel)
    bp = test_pegel["Monat"].iloc[pettitt_test(test_pegel).cp]
    print("\nPettitt test:", res, "\nBruchpunkt:", bp)
    plot_breakpoint(test_pegel, res)