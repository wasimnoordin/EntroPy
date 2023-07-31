# Equal to "quants.py" from FinQuant

""" The module provides functions to compute quantities relevant to financial
portfolios, e.g. a weighted average, which is the expected value/return, a
weighted standard deviation (volatility), and the Sharpe ratio.
"""

import pandas
import numpy
import scipy.stats

def calculate_stratified_average(avg_asset_value, proportion_of_investment):

    """Calculates stratified averages/expected returns of a given portfolio
    
    Arguments: 
    - avg_asset_value: numpy array or pandas series of individual assets in the portfolio
    - portion_of_investment: numpy array or pandas series of proportionality of each investment
    
    Return:
    - st_average: numpy array or pandas series of XYZ
    """ 

    #Check input types
    for arg, arg_name in [(avg_asset_value, 'avg_asset_value'), (proportion_of_investment, 'portion_of_investment')]:
        if not isinstance(arg, (pandas.Series, numpy.ndarray)):
            raise ValueError(f"{arg_name} must be a numpy.ndarray or pandas.Series")

    # Compute and return the stratified average
    st_average = numpy.sum(avg_asset_value * proportion_of_investment)
    return st_average

def calculate_portfolio_volatility(dispersion_matrix, proportion_of_investment):
    """
    Calculates the volatility of a portfolio given a dispersion matrix and each asset's relative magnitude.

    Arguments:
    - dispersion_matrix: numpy array or pandas series of the portfolio's dispersion matrix
    - proportion_of_investment: numpy array or pandas series of proportionality of each investment

    Returns:
    - portfolio_volatility: 
    """

    # Map variable names to their values
    volatility_variables = {'proportion_of_investment': proportion_of_investment, 'dispersion_matrix': dispersion_matrix}

    # Check input types
    for var_name, var_value in volatility_variables.items():
        if not isinstance(var_value, (pandas.Series, numpy.ndarray, pandas.DataFrame)):
            raise TypeError(f"{var_name} must be a pandas.Series, numpy.ndarray, or pandas.DataFrame")

    # Ensure coefficients are a numpy array
    coefficients = numpy.array(proportion_of_investment)

    # Ensure dispersion matrix is a numpy array
    disp_matrix = numpy.array(dispersion_matrix)

    # Calculate portfolio variance
    portfolio_variance = coefficients @ disp_matrix @ coefficients.T

    # Calculate portfolio volatility
    portfolio_volatility = numpy.sqrt(portfolio_variance)
    return portfolio_volatility

def calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_ROR=0.005427):
    """
    Calculates the Sharpe Ratio of a portfolio.

    Arguments:
    - portfolio_return: Numeric, predicted return of the portfolio.
    - portfolio_volatility: Numeric, volatility (or standard deviation) of the portfolio.
    - risk_free_ROR (optional): Numeric, risk-free rate of return. Obtained as 5.427% on 28/07/23 
                                from https://www.marketwatch.com/investing/bond/tmubmusd03m?countrycode=bx

    Returns:
    - sharpe_ratio: Numeric, Sharpe Ratio of the portfolio.
    """
    def validate_sharpe_inputs(*args):
        if not all(isinstance(x, (int, float, numpy.number)) for x in args):
            raise TypeError("All inputs must be numeric.")

    validate_sharpe_inputs(portfolio_return, portfolio_volatility, risk_free_ROR)

    excess_return = portfolio_return - risk_free_ROR
    sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility else 'inf'
    return sharpe_ratio

def calculate_VaR(asset_total, asset_average, asset_volatility, confidence_lvl=0.95):
    """
    Calculates the Value at Risk (VaR) of an a given asset.

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
    if not all(isinstance(x, (int, float, numpy.number)) for x in [asset_total, asset_average, asset_volatility, confidence_lvl]):
        raise TypeError("All inputs must be numeric.")
    
    # Check if conf_level is within range
    if not 0 < confidence_lvl < 1:
        raise ValueError("Confidence level should fall within the range of 0 to 1.")

    # Calculate the inverse of the cumulative distribution function
    inverse_cdf_VaR = scipy.stats.norm.ppf(1 - confidence_lvl)

    # Calculate the difference between the mean return and the product of the standard deviation and the inverse cdf
    diff_VaR = asset_average - asset_volatility * inverse_cdf_VaR

    # Calculate Value at Risk by multiplying the investment with the difference
    asset_VaR = asset_total * diff_VaR
    return asset_VaR

def calculate_annualisation_of_measures(proportion_of_investment, avg_asset_value, disp_matrix, risk_free_ROR=0.005427, regular_trading_days=252):
    """
    Calculates and returns the expected yearly return, volatility, and Sharpe Ratio of a portfolio.

    Parameters:
    - avg_asset_value: numpy array or pandas series of individual assets in the portfolio
    - portion_of_investment: numpy array or pandas series of proportionality of each investment
    - dispersion_matrix: numpy array or pandas series of the portfolio's dispersion matrix.
    - risk_free_ROR (optional): The risk-free rate of return. Obtained as 5.427% on 28/07/23 
                                from https://www.marketwatch.com/investing/bond/tmubmusd03m?countrycode=bx
    - regular_trading_days (optional): Numeric, average number of regular trading days in a year. 
                                Setpoint: 252 days. Taken from: https://en.wikipedia.org/wiki/Trading_day

    Returns:
    - annualised_returns: Tuple, yearly {return, volatility, Sharpe Ratio} of the portfolio.
    """

    # Assert that trading_days_per_year is an integer
    try:
        assert type(regular_trading_days) == int
    except AssertionError:
        raise TypeError("The argument regular_trading_days should be of type int.")

    # Calculate annualised return
    return_factor = regular_trading_days
    average_return = calculate_stratified_average(avg_asset_value, proportion_of_investment)
    annualised_return = return_factor * average_return

    # Calculate annualised volatility
    volatility_factor = numpy.sqrt(regular_trading_days)
    pfolio_volatility = calculate_portfolio_volatility(disp_matrix, proportion_of_investment)
    annualised_volatility = pfolio_volatility * volatility_factor

    # Calculate the annualised Sharpe Ratio
    sharpe_ratio = None
    try:
        sharpe_ratio = calculate_sharpe_ratio(annualised_return, annualised_volatility, risk_free_ROR)
    except ZeroDivisionError:
        print("Warning: Division by zero encountered while calculating Sharpe Ratio. Setting Sharpe Ratio to infinity.")
        sharpe_ratio = float('inf')

    # Return combined annualised output tuple
    annualised_measures = annualised_return, annualised_volatility, sharpe_ratio
    return annualised_measures