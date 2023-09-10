import numpy
import pytest

from src.measures_common import (
    calculate_annualisation_of_measures,
    calculate_stratified_average,
    calculate_portfolio_volatility,
)

def test_calculate_stratified_average():
    """
    Test the calculate_stratified_average function.
    
    This test checks the function's output against expected values for both 
    simple and random inputs.
    """
    # Simple test case
    averages = numpy.array([1])
    allocations = numpy.array([1])
    assert calculate_stratified_average(averages, allocations) == averages.mean()
    
    # Random test case
    averages = numpy.random.rand(10)
    allocations = numpy.random.rand(10)
    forecast = numpy.sum(averages * allocations)
    assert calculate_stratified_average(averages, allocations) == pytest.approx(forecast, 1e-15)

def test_calculate_portfolio_volatility():
    """
    Test the calculate_portfolio_volatility function.
    
    This test checks the function's output against an expected value using random inputs.
    """
    x = numpy.random.rand(10)
    y = numpy.random.rand(10)
    disp_mat = numpy.cov(x, y)
    allocation = numpy.ones(2)
    forecast = numpy.sqrt(numpy.dot(allocation.T, numpy.dot(disp_mat, allocation)))
    assert calculate_portfolio_volatility(disp_mat, allocation) == pytest.approx(forecast, 1e-15)

def test_calculate_annualisation_of_measures():
    """
    Test the calculate_annualisation_of_measures function.
    
    This test checks that all values in the function's output are either integers or floats.
    """
    x = numpy.random.rand(10)
    y = numpy.random.rand(10)
    Sigma = numpy.cov(x, y)
    weights = numpy.ones(2)
    mean_returns = numpy.array([x.mean(), y.mean()])
    results = calculate_annualisation_of_measures(weights, mean_returns, Sigma, 0, 252)
    assert all(isinstance(val, (int, float)) for val in results)

# The following tests check that the functions raise appropriate errors for invalid inputs.

def test_stratified_average_invalid_input():
    """
    Test the calculate_stratified_average function with invalid inputs.
    """
    with pytest.raises(ValueError):
        calculate_stratified_average("string", numpy.array([1, 2]))
    with pytest.raises(ValueError):
        calculate_stratified_average(numpy.array([1, 2]), "string")

def test_portfolio_volatility_invalid_input():
    """
    Test the calculate_portfolio_volatility function with invalid inputs.
    """
    with pytest.raises(TypeError):
        calculate_portfolio_volatility("string", numpy.array([1, 2]))



def test_annualisation_of_measures_invalid_input():
    """
    Test the calculate_annualisation_of_measures function with invalid inputs.
    """
    x = numpy.array([1, 2, 3])
    y = numpy.array([9, 8, 7])
    Sigma = numpy.cov(x, y)
    with pytest.raises(ValueError):
        calculate_annualisation_of_measures("string", y, Sigma, 0, 252)
    with pytest.raises(ValueError):
        calculate_annualisation_of_measures(x, "string", Sigma, 0, 252)
    with pytest.raises(TypeError):
        calculate_annualisation_of_measures(x, y, "string", 0, 252)
