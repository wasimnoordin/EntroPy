import numpy

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
    
    # Internal function to validate the numeric nature of inputs
    def validate_sharpe_inputs(*args):
        if not all(isinstance(x, (int, float, numpy.number)) for x in args):
            raise TypeError("All inputs must be numeric.")

    # Validate the inputs
    validate_sharpe_inputs(portfolio_return, portfolio_volatility, risk_free_ROR)

    # Calculate the excess return over the risk-free rate
    excess_return = portfolio_return - risk_free_ROR
    # Calculate Sharpe ratio; if volatility is zero, return 'inf' (infinity)
    sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility else 'inf'
    return sharpe_ratio

def calculate_sortino_ratio(forecast_revenue, downside_risk, risk_free_ROR=0.005427):
    """
    Calculates the Sortino Ratio of a portfolio.

    Arguments:
    - forecast_revenue: Numeric, predicted return of the portfolio.
    - downside_risk: Numeric, downside risk of the portfolio.
    - risk_free_ROR (optional): Numeric, risk-free rate of return.

    Returns:
    - sortino_ratio: Numeric, Sortino Ratio of the portfolio.
    """
    
    try:
        # Validate inputs
        for value, name in zip([forecast_revenue, downside_risk, risk_free_ROR], ["exp_return", "downside_risk", "risk_free_ROR"]):
            _validate_sortino_input(value, name)

        # Check for zero downside risk
        if downside_risk == 0:
            return numpy.nan

        # Calculate the excess return over the risk-free rate
        excess_return = forecast_revenue - risk_free_ROR
        # Calculate Sortino ratio
        sortino_ratio = excess_return / downside_risk

        return sortino_ratio
    except Exception as e:
        print(f"An error occurred while calculating the Sortino Ratio: {str(e)}")
        return None
    
def _validate_sortino_input(value, name):
    """
    Validates the inputs for the Sortino ratio calculation.

    Arguments:
    - value: Numeric, the value to be validated.
    - name: String, name of the variable for error messaging.

    Raises:
    - ValueError: If the value is not an integer or float.
    """
    
    if not isinstance(value, (int, float, numpy.integer, numpy.floating)):
        raise ValueError(f"{name} is expected to be an integer or float.")
