from types import FunctionType
import pandas as pd
import numpy as np
import pyhomogeneity as hg


def double_sum(test_gauge: list[float], ref_gauge: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Returns a list of double sums."""
    sum_test = np.cumsum(test_gauge)
    sum_ref = np.cumsum(ref_gauge)
    return sum_test, sum_ref

def pettitt_test(df: pd.DataFrame, var: str = "Durchfluss_m3s") -> FunctionType:
    """Test for inhomogeneity using the Pettitt test."""
    return hg.pettitt_test(x=df[var], alpha=0.05)
