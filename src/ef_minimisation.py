import numpy
import pandas
from src.measures import calculate_annualisation_of_measures

def calculate_annualized_volatility(proportion_of_investment, avg_revenue, disp_matrix):
    """Calculates annualized portfolio volatility.

    Parameters:
     -- portion_of_investment: numpy array of allocation of each investment
     -- avg_revenue: numpy array or pandas series of average asset returns
     -- dis_matrix: numpy array or pandas series of dispersion matrix alligned with returns

    Returns:
     -- annualised volatility: annualised volatility measure of the portfolio
    """

    # Validate the input
    if not isinstance(proportion_of_investment, (list, numpy.ndarray)) or not \
                        isinstance(avg_revenue, (list, numpy.ndarray, pandas.Series)) or not \
                              isinstance(disp_matrix, (pandas.DataFrame, numpy.ndarray)):
        raise TypeError("Invalid input types for proportions, average revenue, or dispersion matrix.")

    if len(proportion_of_investment) != len(avg_revenue) or len(proportion_of_investment) != disp_matrix.shape[0]:
        raise ValueError("Mismatch in dimensions between proportions, average revenue, and dispersion matrix.")

    # Normalize the proportions to ensure they sum to 1
    proportion_of_investment = numpy.array(proportion_of_investment)
    proportion_of_investment /= numpy.sum(proportion_of_investment)

    # Call the calculate_annualisation_of_measures function to get the annualized measures
    _, annualised_volatility, _ = calculate_annualisation_of_measures(proportion_of_investment, avg_revenue, disp_matrix)

    # Return the annualized volatility
    return annualised_volatility

def calculate_inverse_sharpe_ratio(proportion_of_investment, avg_revenue, disp_matrix, risk_free_ROR):
    """Calculates the negative Sharpe ratio of a portfolio ##REWORD and ELABORATE ON COMMENT
    Parameters:

    Returns:
    """

    # Check if porportions are provided as a list and convert to numpy array
    if isinstance(proportion_of_investment, list):
        proportion_of_investment = numpy.array(proportion_of_investment)

    # Check if average asset revenue is provided as a list and convert to pandas Series
    if isinstance(avg_revenue, list):
        avg_revenue = pandas.Series(avg_revenue)

    # Check if dispersion matrix is provided as a numpy array and convert to pandas DataFrame
    if isinstance(disp_matrix, numpy.ndarray):
        disp_matrix = pandas.DataFrame(disp_matrix)

    # Validate the dimensions
    if len(proportion_of_investment) != len(avg_revenue) or len(proportion_of_investment) != disp_matrix.shape[0]:
        raise ValueError("Mismatch in dimensions between proportions, average revenue, and dispersion matrix.")

    # Validate the risk-free rate of return
    if not (0 <= risk_free_ROR <= 1):
        raise ValueError("Risk-free rate must be between 0 and 1.")

    # Normalize the proportion / allocations
    proportion_of_investment = proportion_of_investment / proportion_of_investment.sum()

    # Get the Sharpe ratio from the calculate_annualisation_of_measures function
    _, _, min_sharpe_ratio = calculate_annualisation_of_measures(proportion_of_investment, avg_revenue, disp_matrix, risk_free_ROR)

    # Return the inverse Sharpe ratio
    return -(min_sharpe_ratio)

def portfolio_return(proportion_of_investment, avg_revenue, disp_matrix):
    """Calculates the expected annualised return of a portfolio

    Parameters:

    Returns:
    """

    # Validate the input types for proportion_of_investment, avg_revenue, and disp_matrix
    if not isinstance(proportion_of_investment, (list, numpy.ndarray)) or not \
        isinstance(avg_revenue, (list, numpy.ndarray, pandas.Series)) or not \
            isinstance(disp_matrix, (pandas.DataFrame, numpy.ndarray)):
        raise TypeError("Invalid input types for proportions, average revenue, or dispersion matrix.")

    # Validate the dimensions to ensure they match between proportion_of_investment, avg_revenue, and disp_matrix
    if len(proportion_of_investment) != len(avg_revenue) or len(proportion_of_investment) != disp_matrix.shape[0]:
        raise ValueError("Mismatch in dimensions between proportions, average revenue, and dispersion matrix.")

    # Normalize the proportions to ensure they sum to 1, making them valid for the portfolio
    proportion_of_investment = numpy.array(proportion_of_investment)
    proportion_of_investment /= numpy.sum(proportion_of_investment)

    # Call the calculate_annualisation_of_measures function to get the annualised return
    # The function is expected to return a tuple, and we are interested in the first element (annualised return)
    annualised_return, _, _ = calculate_annualisation_of_measures(proportion_of_investment, avg_revenue, disp_matrix)

    # Return the calculated annualised return
    return annualised_return
