# Import necessary libraries
import numpy
import pandas
import pytest


from src.measures_risk import (
    calculate_value_at_risk,
    calculate_downside_risk,
)

def test_calculate_value_at_risk():
    """
    Test the calculate_value_at_risk function.
    
    This test checks the type of the result and ensures that the value at risk 
    cannot exceed the initial investment.
    """
    # Calculate value at risk with given parameters
    result = calculate_value_at_risk(1e2, 0.5, 0.25, 0.95)
    
    # Assert that the result is either an integer or a float
    assert isinstance(result, (int, float))
    
    # Assert that the result is less than or equal to the initial investment
    assert result <= 1e2

def test_calculate_downside_risk():
    """
    Test the calculate_downside_risk function.
    
    This test checks the type of the result and ensures that the downside risk 
    is non-negative.
    """
    # Create a sample DataFrame with random data
    data = pandas.DataFrame({"1": numpy.random.rand(10), "2": numpy.random.rand(10)})
    
    # Define weights and risk-free rate
    weights = numpy.array([0.5, 0.5])
    rf_rate = 0.005427
    
    # Calculate downside risk with given parameters
    dr = calculate_downside_risk(data, weights, rf_rate)
    
    # Assert that the result is either an integer or a float
    assert isinstance(dr, (int, float))
    
    # Assert that the downside risk is non-negative
    assert dr >= 0

def test_value_at_risk_invalid_input():
    """
    Test the calculate_value_at_risk function with invalid inputs.
    
    This test checks that the function raises a TypeError when provided with 
    incorrect input types.
    """
    # Assert that a TypeError is raised when the function is called with string inputs
    with pytest.raises(TypeError):
        calculate_value_at_risk("string", 0.5, 0.25, 0.95)
    with pytest.raises(TypeError):
        calculate_value_at_risk(1e2, "string", 0.25, 0.95)
    with pytest.raises(TypeError):
        calculate_value_at_risk(1e2, 0.5, "string", 0.95)
    with pytest.raises(TypeError):
        calculate_value_at_risk(1e2, 0.5, 0.25, "string")

def test_downside_risk_invalid_input(capfd):
    """
    Test the calculate_downside_risk function with mismatched input dimensions.
    
    This test checks that the function prints an error message when provided with 
    a DataFrame and weights of different dimensions.
    """
    # Call the function with mismatched input dimensions
    calculate_downside_risk(pandas.DataFrame({"1": numpy.random.rand(10)}), numpy.array([0.5, 0.5]), 0.005)
    
    # Capture the printed output
    captured = capfd.readouterr()
    
    # Assert that the expected error message is in the captured output
    assert "Input DataFrame and weights must have matching dimensions." in captured.out
