import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '/home/lena/Dokumente/FGB/StochastischeSpeicherbemessung/')

import utils.stats as st


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

def test_primary_stats(first_4_rows: pd.DataFrame):
    
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

def test_FSA():
    # TODO
    assert True