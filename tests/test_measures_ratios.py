import numpy
import pytest

from src.measures_ratios import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
def test_calculate_sharpe_ratio():
    """
    Test the calculate_sharpe_ratio function.
    
    This test checks the function's output against an expected value using predefined inputs.
    """
    rf = 0.02
    mean_return = 0.08
    volatility = 0.15
    expected = (mean_return - rf) / volatility
    assert calculate_sharpe_ratio(mean_return, volatility, rf) == pytest.approx(expected, 1e-15)

def test_calculate_sortino_ratio():
    """
    Test the calculate_sortino_ratio function.
    
    This test checks the function's output against expected values for both 
    a zero downside standard deviation and a non-zero value.
    """
    rf = 0.02
    mean_return = 0.08
    downside_std = 0.15
    if downside_std == 0:
        assert calculate_sortino_ratio(mean_return, downside_std, rf) is numpy.NaN
    else:
        expected = (mean_return - rf) / downside_std
        assert calculate_sortino_ratio(mean_return, downside_std, rf) == pytest.approx(expected, 1e-15)

def test_sharpe_ratio_invalid_input():
    """
    Test the calculate_sharpe_ratio function with invalid inputs.
    """
    with pytest.raises(TypeError):
        calculate_sharpe_ratio("string", 0.2, 0.02)

def test_sortino_ratio_edge_cases():
    """
    Test the calculate_sortino_ratio function for edge cases.
    
    Specifically, this test checks the function's behavior when the downside standard deviation is zero.
    """
    rf = 0.02
    mean_return = 0.08
    assert calculate_sortino_ratio(mean_return, 0, rf) is numpy.NaN