import numpy as np


def rel_yearly(data: list[float]):
    """Return reliability metrics for yearly values."""
    data = np.array(data).reshape((40, 12))
    n_years = 0
    for i in range(len(data)):
        if np.sum(data[i]) > 0:
            n_years += 1
    
    return round(1 - n_years / 40, 3)

def rel_monthly(data: list[float]):
    """Return reliability metrics for monthly values."""
    n_months = 0
    for i in range(len(data)):
        if data[i] > 0:
            n_months += 1
    return round(1 - n_months / (40*12), 3)

def rel_amount(data: list[float], soll_abgabe: list[float]):
    """Return the amout of deficit and overflow from Soll-Abgabe."""
    return abs(np.sum(data)) / np.sum(soll_abgabe)