import os
import sys
import inspect
import pytest
import numpy as np
import pandas as pd


from code.utils.primary_stats import sample_number, earliest_date, latest_date, \
    years, hyd_years, min_val, max_val, min_val_month, max_val_month, \
    first_central_moment, second_central_moment, third_central_moment, \
    fourth_central_moment, standard_deviation_biased, standard_deviation_unbiased, \
    skewness_biased, skewness_unbiased, kurtosis_biased, kurtosis_unbiased, \
    quartile, iqr
  

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
    
    assert sample_number(first_4_rows) == 4
    assert earliest_date(first_4_rows) == pd.Timestamp("1959-11-01")
    assert latest_date(first_4_rows) == pd.Timestamp("1960-02-01")
    
    assert (years(first_4_rows) == np.array([1959, 1960])).all()
    assert (hyd_years(first_4_rows) == np.array([1960])).all()
    
    assert min_val(first_4_rows) == 0
    assert max_val(first_4_rows) == 2
    assert min_val_month(first_4_rows) == "11/1959"
    assert max_val_month(first_4_rows) == "02/1960"
    
    assert first_central_moment(first_4_rows) == 1.0
    assert second_central_moment(first_4_rows) == 0.5
    assert third_central_moment(first_4_rows) == 0.0
    assert fourth_central_moment(first_4_rows) == 0.5
    
    assert round(standard_deviation_biased(first_4_rows), 4) == 0.7071
    assert round(standard_deviation_unbiased(first_4_rows), 4) == 0.8165
    
    assert skewness_biased(first_4_rows) == 0.0
    assert skewness_unbiased(first_4_rows) == 0.0
    
    assert round(kurtosis_biased(first_4_rows), 1) == 5.0
    assert round(kurtosis_unbiased(first_4_rows), 1) == 20.0
    
    assert quartile(first_4_rows, which="Q1") == 1.0
    assert quartile(first_4_rows, which="Q2") == 1.0
    assert quartile(first_4_rows, which="Q3") == 1.0
    assert iqr(first_4_rows) == 0.0


def test_trend_analysis():
    # TODO
    assert True
