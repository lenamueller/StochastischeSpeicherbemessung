import os
import sys
import inspect
import pytest
import numpy as np
import pandas as pd

from code.utils.primary_stats import sample_number


# from code import sample_number, earliest_date, latest_date, \
#     years, hyd_years, min_val, max_val, min_val_month, max_val_month, \
#     first_central_moment
  
  

# @pytest.fixture
# def random_arr():
#     return np.random.rand(4, 3)

# @pytest.fixture
# def random_df():
#     return pd.DataFrame(np.random.rand(4, 3), columns=["Monat", "Durchfluss_m3s", "Datum"])


def first_4_rows():
    return pd.DataFrame({
        "Monat": ["11/1959", "12/1959", "01/1960", "02/1960"],
        "Durchfluss_m3s": [1, 0, 2, 1],
        "Datum": [
            pd.Timestamp("1959-11-01"),
            pd.Timestamp("1959-12-01"),
            pd.Timestamp("1960-01-01"),
            pd.Timestamp("1960-02-01")
            ]
    })


def test_mean_monthly():
    test_data = first_4_rows()
    
    assert sample_number(test_data) == 4
    # assert earliest_date(test_data) == pd.Timestamp("1959-11-01")
    # assert latest_date(test_data) == pd.Timestamp("1960-02-01")
    # assert (years(test_data) == np.array([1959, 1960])).all()
    # assert (hyd_years(test_data) == np.array([1960])).all()
    # assert min_val(test_data) == 0
    # assert max_val(test_data) == 2
    # assert min_val_month(test_data) == "12/1959"
    # assert max_val_month(test_data) == "01/1960"
    # assert first_central_moment(test_data) == 1.0

    # TODO
    # assert second_central_moment(test_data) == 0.0
    # assert third_central_moment(test_data) == 0.0
    # assert fourth_central_moment(test_data) == 0.0
    
    # assert standard_deviation_biased(test_data) == 0.0
    # assert standard_deviation_unbiased(test_data) == 0.0
    # assert skewness_biased(test_data) == 0.0
    # assert skewness_unbiased(test_data) == 0.0
    # assert kurtosis_biased(test_data) == 0
    # assert kurtosis_unbiased(test_data) == 0
    # assert quartile(test_data, 0.25) == 0
    # assert quartile(test_data, 0.5) == 0
    # assert quartile(test_data, 0.75) == 0
    # assert iqr(test_data) == 0
    