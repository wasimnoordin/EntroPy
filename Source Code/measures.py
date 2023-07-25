# Equal to "quants.py" from FinQuant

""" The module provides functions to compute quantities relevant to financial
portfolios, e.g. a weighted average, which is the expected value/return, a
weighted standard deviation (volatility), and the Sharpe ratio.
"""

import pandas
import numpy
import scipy.stats

def calculate_stratified_average(avg_asset_value, portion_of_investment):

    """Calculates stratified averages/expected returns of a given portfolio
    
    Parameters: 
    - avg_asset_value: numpy array or pandas series of individual assets in the portfolio
    - portion_of_investment: numpy array or pandas series of proportionality of each investment
    """ 

    #Check input types
    for arg, arg_name in [(avg_asset_value, 'avg_asset_value'), (portion_of_investment, 'portion_of_investment')]:
        if not isinstance(arg, (pandas.Series, numpy.ndarray)):
            raise ValueError(f"{arg_name} must be a pandas.Series or numpy.ndarray")

    # Compute and return the stratified average
    st_average = numpy.sum(avg_asset_value * portion_of_investment)
    return st_average