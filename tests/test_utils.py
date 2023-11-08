import pytest
import numpy as np
import pandas as pd

import code.utils.statistics as st
  

@pytest.fixture
def random_arr():
    return np.random.rand(4, 3)

@pytest.fixture
def random_df():
    return pd.DataFrame(
        np.random.rand(4, 3),
        columns=["Monat", "Durchfluss_m3s", "Datum"]
        )

@pytest.fixture
def first_4_rows():
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



def test_binned_stats():
    # TODO
    assert True

def test_consistency_check():
    # TODO
    assert True

def test_data_structures():
    # TODO
    assert True

def test_fft_analysis():
    # TODO
    assert True

def test_hydrological_values():
    # TODO
    assert True

def test_primary_stats(first_4_rows: pd.DataFrame):
    
    assert st.sample_number(first_4_rows) == 4
    assert st.earliest_date(first_4_rows) == pd.Timestamp("1959-11-01")
    assert st.latest_date(first_4_rows) == pd.Timestamp("1960-02-01")
    assert (st.years(first_4_rows) == np.array([1959, 1960])).all()
    assert (st.hyd_years(first_4_rows) == np.array([1960])).all()
    assert st.min_val(first_4_rows) == 0
    assert st.max_val(first_4_rows) == 2
    assert st.min_val_month(first_4_rows) == "11/1959"
    assert st.max_val_month(first_4_rows) == "02/1960"
    assert st.first_central_moment(first_4_rows) == 1.0
    assert st.second_central_moment(first_4_rows) == 0.5
    assert st.third_central_moment(first_4_rows) == 0.0
    assert st.fourth_central_moment(first_4_rows) == 0.5
    assert round(st.standard_deviation_biased(first_4_rows), 4) == 0.7071
    assert round(st.standard_deviation_unbiased(first_4_rows), 4) == 0.8165
    assert round(st.skewness_biased(first_4_rows), 3) == 0.0
    assert round(st.skewness_unbiased(first_4_rows), 3) == 0.0
    assert round(st.kurtosis_biased(first_4_rows), 1) == 5.0
    assert round(st.kurtosis_unbiased(first_4_rows), 1) == 20.0
    assert st.quantiles(first_4_rows, q=0.25) == 1.0
    assert st.quantiles(first_4_rows, q=0.50) == 1.0
    assert st.quantiles(first_4_rows, q=0.75) == 1.0
    assert st.iqr(first_4_rows) == 0.0


def test_trend_analysis():
    # TODO
    assert True
