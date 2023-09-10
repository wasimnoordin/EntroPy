import pytest
import numpy

from src.mef_minimisation import (
    calculate_annualized_volatility, 
    calculate_inverse_sharpe_ratio, 
    calculate_annualised_return
)

# Sample data for testing
avg_revenue = numpy.array([0.05, 0.06])
proportions = numpy.array([0.5, 0.5])
disp_matrix = numpy.array([[0.1, 0.03], [0.03, 0.12]])
risk_free_ROR = 0.05

# Testing the functionality of the `calculate_annualized_volatility` function

# 1. Test for valid inputs to the function.
def test_calculate_annualized_volatility():
    # Calling the function with valid inputs
    vol = calculate_annualized_volatility(proportions, avg_revenue, disp_matrix)
    
    # Asserting that the returned value is of type float or numpy.float64
    assert isinstance(vol, (float, numpy.float64))
    
    # Asserting that the volatility value is non-negative
    assert vol >= 0

# Test for invalid inputs to the function.
def test_calculate_annualized_volatility_invalid_input():
    # Expecting a TypeError when providing a string value for average revenue
    with pytest.raises(TypeError):
        calculate_annualized_volatility([0.5, 0.5], "0.05", disp_matrix)
    
    # Expecting a ValueError when providing mismatched dimensions for the inputs
    with pytest.raises(ValueError):
        calculate_annualized_volatility([0.5], avg_revenue, disp_matrix)

# Testing the functionality of the `calculate_inverse_sharpe_ratio` function

# 2. Test for valid inputs to the function.
def test_calculate_inverse_sharpe_ratio():
    # Calling the function with valid inputs
    neg_sharpe = calculate_inverse_sharpe_ratio(proportions, avg_revenue, disp_matrix, risk_free_ROR)
    
    # Asserting that the returned value is of type float or numpy.float64
    assert isinstance(neg_sharpe, (float, numpy.float64))

# Test for invalid inputs to the function.
def test_calculate_inverse_sharpe_ratio_invalid_input():
    # Expecting a ValueError when providing a risk-free rate of return greater than 1
    with pytest.raises(ValueError):
        calculate_inverse_sharpe_ratio(proportions, avg_revenue, disp_matrix, 1.5)
    
    # Expecting a ValueError when providing mismatched dimensions for the inputs
    with pytest.raises(ValueError):
        calculate_inverse_sharpe_ratio([0.5], avg_revenue, disp_matrix, risk_free_ROR)

# Testing the functionality of the `calculate_annualised_return` function

# 3. Test for valid inputs to the function.
def test_calculate_annualised_return():
    # Calling the function with valid inputs
    ann_return = calculate_annualised_return(proportions, avg_revenue, disp_matrix)
    
    # Asserting that the returned value is of type float or numpy.float64
    assert isinstance(ann_return, (float, numpy.float64))

# Test for invalid inputs to the function.
def test_calculate_annualised_return_invalid_input():
    # Expecting a TypeError when providing a string value for average revenue
    with pytest.raises(TypeError):
        calculate_annualised_return([0.5, 0.5], "0.05", disp_matrix)
    
    # Expecting a ValueError when providing mismatched dimensions for the inputs
    with pytest.raises(ValueError):
        calculate_annualised_return([0.5], avg_revenue, disp_matrix)
