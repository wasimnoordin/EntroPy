import numpy
import pandas
import scipy.stats

from src.measures_common import calculate_daily_return_proportioned


def calculate_value_at_risk(asset_total, asset_average, asset_volatility, confidence_interval=0.95):
    """
    Calculates the Value at Risk of an a given asset.

    Arguments:
    - asset_total: Numeric, absolute value of all assets. 
    - asset_average: Numeric, asset average return.
    - asset_volatility: Numeric, asset standard deviation.
    - confidence_lvl (optional): Numeric, confidence level for the VaR calculation. 
                                Setpoint: 0.95, as used by JPM, GMS, and other IB/HF.

    Returns:
    - asset_VaR: Numeric, value at Risk of the asset.
    """

    # Check if inputs are numeric
    if not all(isinstance(x, (int, float, numpy.number)) for x in [asset_total, asset_average, asset_volatility, confidence_interval]):
        raise TypeError("All inputs must be numeric.")
    
    # Check if conf_level is within range
    if not 0 < confidence_interval < 1:
        raise ValueError("Confidence level should fall within the range of 0 to 1.")

    # Calculate the inverse of the cumulative distribution function
    inverse_cdf_value_at_risk = scipy.stats.norm.ppf(1 - confidence_interval)

    # Calculate the difference between the mean return and the product of the standard deviation and the inverse cdf
    diff_value_at_risk = asset_average - asset_volatility * inverse_cdf_value_at_risk

    # Calculate Value at Risk by multiplying the investment with the difference
    asset_value_at_risk = asset_total * diff_value_at_risk
    return asset_value_at_risk

def calculate_downside_risk(input_stock_prices: pandas.DataFrame, proportion, risk_free_ROR=0.005427) -> float:
    """
    Calculate the downside risk of a portfolio given stock prices, proportions, and a risk-free rate of return.
    
    Parameters:
    - input_stock_prices: DataFrame containing stock prices.
    - proportion: Weights of the assets in the portfolio.
    - risk_free_ROR: Risk-free rate of return. Default is 0.005427.
    
    Returns:
    - float: The calculated downside risk.
    """
    try:
        # Validate inputs
        _validate_downside_input(input_stock_prices, pandas.DataFrame, "Data must be a Pandas DataFrame.")
        _validate_downside_input(proportion, (pandas.Series, numpy.ndarray), "Weights must be a pandas Series or numpy ndarray.")
        _validate_downside_input(risk_free_ROR, (int, float, numpy.integer, numpy.floating), "Risk-free rate must be an integer or float.")

        # Compute weighted daily mean returns
        wtd_daily_mean = calculate_daily_return_proportioned(input_stock_prices, proportion)

        # Compute downside risk
        downside_risk = _compute_downside_risk(wtd_daily_mean, risk_free_ROR)

        return downside_risk
    except Exception as e:
        print(f"An error occurred while calculating downside risk: {str(e)}")
        return None

def _validate_downside_input(value, expected_type, error_message):
    """
    Validate the input values for type consistency.
    
    Parameters:
    - value: The input value to validate.
    - expected_type: The expected type of the input value.
    - error_message: The error message to raise if validation fails.
    """
    if not isinstance(value, expected_type):
        raise ValueError(error_message)

def _compute_downside_risk(wtd_daily_mean, risk_free_ROR):
    """
    Compute the downside risk of a portfolio.
    
    Parameters:
    - wtd_daily_mean: Weighted daily mean returns.
    - risk_free_ROR: Risk-free rate of return.
    
    Returns:
    - float: The calculated downside risk.
    """
    # Calculate the differences between the weighted daily mean and the risk-free rate of return
    differences = wtd_daily_mean - risk_free_ROR

    # Identify the negative returns (returns below the risk-free rate)
    negative_returns = numpy.minimum(0, differences)

    # Square the negative returns
    squared_negative_returns = negative_returns ** 2

    # Calculate the mean of the squared negative returns
    mean_squared_negative_returns = numpy.mean(squared_negative_returns)

    # Calculate the square root of the mean of the squared negative returns
    downside_risk = numpy.sqrt(mean_squared_negative_returns)

    # Return the downside risk
    return downside_risk
