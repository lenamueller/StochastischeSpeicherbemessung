import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '/home/lena/Dokumente/FGB/StochastischeSpeicherbemessung/')

import utils.stats as st
from O3_Speicherbemessung.sequent_peak_algorithm import calc_maxima, calc_minima, \
    calc_capacity, calc_storage_simulation


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def first_4_rows() -> pd.DataFrame:
    """Create a discharge time series."""
    return pd.DataFrame({
        "Monat": ["11/1959", "12/1959", "01/1960", "02/1960"],
        "Durchfluss_m3s": [0, 1, 1, 2],
        "Datum": [
            pd.Timestamp("1959-11-01"),
            pd.Timestamp("1959-12-01"),
            pd.Timestamp("1960-01-01"),
            pd.Timestamp("1960-02-01")
            ]
    })

@pytest.fixture
def cum_storage_sample() -> list[float]:
    """Create a cumulative storage time series."""
    return [1.0, 3.0, 2.0, 3.0, 1.0, 6.0, 5.0]

# --------------------------------------------------
# Test stats
# --------------------------------------------------

def test_primary_stats(first_4_rows: pd.DataFrame) -> None:
    """Test primary statistics."""
    
    assert st.sample_number(first_4_rows) == 4
    assert st.earliest_date(first_4_rows) == pd.Timestamp("1959-11-01")
    assert st.latest_date(first_4_rows) == pd.Timestamp("1960-02-01")
    assert (st.years(first_4_rows) == np.array([1959, 1960])).all()
    assert (st.hyd_years(first_4_rows) == np.array([1960])).all()
    assert st.min_val(first_4_rows) == (0, "11/1959")
    assert st.max_val(first_4_rows) == (2, "02/1960")
    assert st.central_moment(df=first_4_rows, nth=1) == 1.0
    assert st.central_moment(df=first_4_rows, nth=2) == 0.5
    assert st.central_moment(df=first_4_rows, nth=3) == 0.0
    assert st.central_moment(df=first_4_rows, nth=4) == 0.5
    assert round(st.standard_deviation(first_4_rows, bias=True), 4) == 0.7071
    assert round(st.standard_deviation(first_4_rows, bias=False), 4) == 0.8165
    assert round(st.skewness(first_4_rows, bias=True), 3) == 0.0
    assert round(st.skewness(first_4_rows, bias=False), 3) == 0.0
    assert round(st.kurtosis(first_4_rows, bias=True), 1) == 5.0
    assert round(st.kurtosis(first_4_rows, bias=False), 1) == 20.0
    assert st.quantiles(first_4_rows, q=0.25) == 1.0
    assert st.quantiles(first_4_rows, q=0.50) == 1.0
    assert st.quantiles(first_4_rows, q=0.75) == 1.0
    assert st.iqr(first_4_rows) == 0.0
    
def test_binned_stats():
    # TODO
    assert True

# --------------------------------------------------
# Test FSA
# --------------------------------------------------

def test_maxima(cum_storage_sample: list[float]) -> None:
    """Test calculation of maxima."""
    assert calc_maxima(cum_storage=cum_storage_sample)[0] == [3.0, 6.0]
    assert calc_maxima(cum_storage=cum_storage_sample)[1] == [1, 5]
    with pytest.raises(AssertionError):
        calc_maxima(cum_storage=[])

def testcalc_minima(cum_storage_sample: list[float]) -> None:
    """Test calculation of minima."""
    assert calc_minima(cum_storage=cum_storage_sample, max_indices=[1, 5])[0] == [1.0]
    assert calc_minima(cum_storage=cum_storage_sample, max_indices=[1, 5])[1] == [4]
    with pytest.raises(AssertionError):
        calc_minima(cum_storage=cum_storage_sample, max_indices=[])
    with pytest.raises(AssertionError):
        calc_minima(cum_storage=[], max_indices=[1, 5])
    
def test_capacity(cum_storage_sample: list[float]) -> None:
    """Test calculation of capacity."""
    cap, cap_min_index, cap_min, cap_max_index, cap_max = calc_capacity(cum_storage_sample)
    
    assert cap == 2.0
    assert cap_min == 1.0
    assert cap_min_index == 4
    assert cap_max == 3.0
    assert cap_max_index == 1

def test_fsa_normal():
    """Test FSA with normal conditions."""
    storage, deficit, overflow, q_out_ist = calc_storage_simulation(
        q_in=                   [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 5.0, 2.0],
        q_out_soll=             [1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0],
        initial_storage=        0.0,    
        max_cap=                7.0,
        q_in_convert=False,
        q_out_soll_convert=False
    )
    assert storage ==           [0.0, 1.0, 3.0, 0.0, 1.0, 3.0, 7.0, 5.0]
    assert deficit ==           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert overflow ==          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert q_out_ist ==         [1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0]

def test_fsa_overflow():    
    """Test FSA with overflow."""
    storage, deficit, overflow, q_out_ist = calc_storage_simulation(
        q_in=                   [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 5.0, 2.0],
        q_out_soll=             [1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0],
        initial_storage=        0.0,    
        max_cap=                5.0,
        q_in_convert=False,
        q_out_soll_convert=False
    )
    assert storage ==           [0.0, 1.0, 3.0, 0.0, 1.0, 3.0, 5.0, 3.0]
    assert deficit ==           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert overflow ==          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]
    assert q_out_ist ==         [1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 3.0, 4.0]

def test_fsa_deficit():
    """Test FSA with deficit."""
    storage, deficit, overflow, q_out_ist = calc_storage_simulation(
        q_in=                   [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 5.0, 2.0],
        q_out_soll=             [1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 4.0],
        initial_storage=        0.0,    
        max_cap=                7.0,
        q_in_convert=False,
        q_out_soll_convert=False
    )
    assert storage ==           [0.0, 1.0, 3.0, 0.0, 1.0, 3.0, 7.0, 5.0]
    assert deficit ==           [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    assert overflow ==          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert q_out_ist ==         [1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0]