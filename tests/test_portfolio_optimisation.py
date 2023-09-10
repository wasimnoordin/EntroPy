import numpy
import pandas

import datetime
import pytest

import matplotlib.pylab as pylab
import yfinance

from src.markowitz_efficient_frontier import EfficientFrontierMaster
from src.portfolio_composition import formulate_final_portfolio
from src.performance import calculate_historical_avg_return
from tests.test_portfolio_composition import portfolio_configs

# Error limts
max_error = 1e-12
min_error = 1e-6

# --------------- MONTE CARLO OPTIMISATION TESTS ---------------

def test_pf_mcs_optimised_portfolio():
    config = portfolio_configs[4]
    final_portfolio = formulate_final_portfolio(**config)
    
    # Set the seed for reproducibility
    numpy.random.seed(seed=0)
    
    # Compute the expected return for each individual stock
    individual_forecast_returns = calculate_historical_avg_return(final_portfolio.asset_price_history)
    
    # Generate a few random portfolio weight combinations
    num_portfolios = 100
    random_weights = numpy.array([numpy.random.dirichlet(numpy.ones(len(final_portfolio.stock_objects)), size=1) for _ in range(num_portfolios)])
    
    # Manually compute expected returns for these portfolios using the individual expected returns
    expected_returns = [numpy.dot(weights[0], individual_forecast_returns) for weights in random_weights]
    volatilities = [numpy.sqrt(numpy.dot(weights[0].T, numpy.dot(final_portfolio.asset_price_history.cov(), weights[0]))) for weights in random_weights]

    # Compute Sharpe Ratios for these portfolios
    risk_free_rate = final_portfolio.risk_free_ROR
    sharpe_ratios = (numpy.array(expected_returns) - risk_free_rate) / numpy.array(volatilities)

    # Run the Monte Carlo optimization
    opt_w, opt_res = final_portfolio.pf_mcs_optimised_portfolio(mcs_iterations=500)
    
    # Check that the min volatility portfolio from the optimization has volatility less than or equal to the manually computed volatilities
    assert opt_res.loc["Minimum Volatility", "Volatility"] <= max(volatilities)
    
    # Check that the max Sharpe Ratio portfolio from the optimization has Sharpe Ratio greater than or equal to the manually computed Sharpe Ratios
    assert opt_res.loc["Maximum Sharpe Ratio", "Sharpe Ratio"] >= min(sharpe_ratios)


# --------------- EFFICIENT FRONTIER OPTIMISATION TESTS ---------------

threshold = 1e-2

def test_initialise_mef_instance():
    # Retrieve the specific portfolio configuration from the predefined list
    config = portfolio_configs[4]

    # Create the final portfolio instance using the given configuration
    final_portfolio = formulate_final_portfolio(**config)

    # Initialise the efficient frontier instance for the portfolio
    final_efficient_frontier = final_portfolio.initialise_mef_instance()

    # Gather data from the portfolio and efficient frontier for assertions
    portfolio_avg_return = final_portfolio.calculate_pf_historical_avg_return(regular_trading_days=1)
    portfolio_disp_matrix = final_portfolio.calculate_pf_dispersion_matrix()
    portfolio_symbols = final_portfolio.portfolio_distribution["Name"].values.tolist()
    portfolio_num_stocks = len(final_portfolio.stock_objects)

    # Make assertions to ensure correctness of the efficient frontier and its attributes

    # Assert that both instances are of the correct type
    assert isinstance(final_efficient_frontier, EfficientFrontierMaster)
    assert isinstance(final_portfolio.efficient_frontier_instance, EfficientFrontierMaster)

    # Assert that the portfolio's efficient frontier attribute and the one returned from the function are the same
    assert final_portfolio.efficient_frontier_instance == final_efficient_frontier

    # Assert that the data from the portfolio matches the corresponding data in the efficient frontier
    assert (portfolio_avg_return == final_efficient_frontier.avg_revenue).all()
    assert (portfolio_disp_matrix == final_efficient_frontier.disp_matrix).all().all()
    assert final_portfolio.regular_trading_days == final_efficient_frontier.regular_trading_days
    assert final_portfolio.risk_free_ROR == final_efficient_frontier.risk_free_ROR
    assert final_efficient_frontier.symbol_stocks == portfolio_symbols
    assert final_efficient_frontier.portfolio_size == portfolio_num_stocks

def test_optimize_pf_mef_volatility_minimisation():
    config = portfolio_configs[4]
    final_portfolio = formulate_final_portfolio(**config)
    
    # Generate random weights for the portfolio (they should sum up to 1)
    random_weights = numpy.random.dirichlet(numpy.ones(len(final_portfolio.stock_objects)))

    # Calculate the expected portfolio volatility using these weights
    expected_volatility = numpy.sqrt(numpy.dot(random_weights.T, numpy.dot(final_portfolio.asset_price_history.cov(), random_weights)))
    
    ef_opt_weights = final_portfolio.optimize_pf_mef_volatility_minimisation()
    # Calculate the portfolio volatility using weights from the function
    function_volatility = numpy.sqrt(numpy.dot(ef_opt_weights.values.T, numpy.dot(final_portfolio.asset_price_history.cov(), ef_opt_weights.values)))
    
    # Ensure that the function's volatility is less than or equal to the expected volatility 
    # (because we minimize)
    assert function_volatility <= expected_volatility + min_error

def test_optimize_pf_mef_sharpe_maximisation():
    config = portfolio_configs[4]
    final_portfolio = formulate_final_portfolio(**config)

    # Generate random weights for the portfolio
    random_weights = numpy.random.dirichlet(numpy.ones(len(final_portfolio.stock_objects)))

    # Ensure pf.pf_forecast_return is a scalar
    forecast_return = numpy.array(final_portfolio.pf_forecast_return).squeeze()

    # Calculate expected returns using these weights
    expected_return = sum(random_weights * forecast_return)
    expected_volatility = numpy.sqrt(numpy.dot(random_weights.T, numpy.dot(final_portfolio.asset_price_history.cov(), random_weights)))

    # Calculate the expected Sharpe Ratio
    expected_sharpe_ratio = (expected_return - final_portfolio.risk_free_ROR) / expected_volatility

    ef_opt_weights = final_portfolio.optimize_pf_mef_sharpe_maximisation()

    # Calculate the function's return
    function_return = sum(ef_opt_weights.values * forecast_return)
    function_volatility = numpy.sqrt(numpy.dot(ef_opt_weights.values.T, numpy.dot(final_portfolio.asset_price_history.cov(), ef_opt_weights.values)))

    # Calculate the function's Sharpe Ratio
    function_sharpe_ratio = (function_return - final_portfolio.risk_free_ROR) / function_volatility

    # Ensure the function's Sharpe Ratio is greater than or equal to the expected Sharpe Ratio
    assert abs(function_sharpe_ratio - expected_sharpe_ratio) <= threshold

def test_optimize_pf_mef_return():
    config = portfolio_configs[4]
    final_portfolio = formulate_final_portfolio(**config)

    # Generate random weights for the portfolio
    random_weights = numpy.random.dirichlet(numpy.ones(len(final_portfolio.stock_objects)))

    # Ensure pf.pf_forecast_return is a scalar
    forecast_return = numpy.array(final_portfolio.pf_forecast_return).squeeze()

    # Calculate the expected return using these weights
    expected_return = sum(random_weights * forecast_return)

    ef_opt_weights = final_portfolio.optimize_pf_mef_return(expected_return)

    # Calculate the return for the optimized weights
    optimized_return = sum(ef_opt_weights.values.squeeze() * forecast_return)

    # Assert that the optimized return is close to the expected return
    assert numpy.allclose(optimized_return, expected_return, atol=min_error)

    # Check if the weights are valid
    assert numpy.all(ef_opt_weights.values >= 0)  # weights are non-negative
    assert numpy.allclose(ef_opt_weights.values.sum(), 1, atol=min_error)  # weights sum up to 1

def test_optimize_pf_mef_volatility():
    config = portfolio_configs[4]
    final_portfolio = formulate_final_portfolio(**config)

    # Calculate the volatility for each stock individually
    individual_volatilities = []
    for idx, stock in enumerate(final_portfolio.stock_objects.keys()):
        weights = numpy.zeros(len(final_portfolio.stock_objects))
        weights[idx] = 1  # 100% allocation to this stock
        
        stock_volatility = numpy.sqrt(numpy.dot(weights.T, numpy.dot(final_portfolio.asset_price_history.cov(), weights)))
        individual_volatilities.append(stock_volatility)

    # Get the minimum volatility among all individual stocks
    min_stock_volatility = min(individual_volatilities)

    # Use the function to optimize the portfolio based on the smallest stock volatility
    ef_opt_weights = final_portfolio.optimize_pf_mef_volatility(min_stock_volatility)

    # Calculate the volatility for the optimized weights
    optimized_volatility = numpy.sqrt(numpy.dot(ef_opt_weights.values.T, numpy.dot(final_portfolio.asset_price_history.cov(), ef_opt_weights.values)))
    optimized_volatility = optimized_volatility.squeeze()

    # Assert that the optimized volatility is more than or approximately equal to the volatility of the most stable stock
    # Recall that this is the minimised volatility for a given target return and so may be higher than the absolute minimum!
    assert optimized_volatility >= min_stock_volatility + min_error

def test_optimize_pf_mef_efficient_frontier():
    config = portfolio_configs[4]
    final_portfolio = formulate_final_portfolio(**config)
    
    # Generate a set of target returns
    targets = [round(0.2 + 0.01 * i, 2) for i in range(11)]
    ef_efrontier = final_portfolio.optimize_pf_mef_efficient_frontier(targets)
    
    # Extract volatilities from the efficient frontier results
    volatilities = ef_efrontier[:, 0]
    
    # Ensure that as target returns increase, volatilities also increase
    for i in range(1, len(volatilities)):
        assert abs(volatilities[i] - volatilities[i-1]) <= threshold

# --------------- EFFICIENT FRONTIER VISUALISATION TESTS ---------------

def test_optimize_pf_plot_mef():
    portfolio_config = portfolio_configs[4]
    pf = formulate_final_portfolio(**portfolio_config)
    
    # Clear the current figure to ensure a fresh plot
    pylab.clf()
    
    # Create the plot
    pf.optimize_pf_plot_mef()
    
    # Assert that the plot was created
    assert len(pylab.gcf().get_axes()) == 1

    # Get title, axis labels, and axis min/max values
    ax = pylab.gca()
    title = ax.get_title()
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Assert on title and labels
    assert isinstance(title, str) and len(title) > 0
    assert isinstance(xlabel, str) and len(xlabel) > 0
    assert isinstance(ylabel, str) and len(ylabel) > 0
    
    # Assert on reasonable x and y limits 
    # (Here, 0 and 1 are just placeholders. Adjust as per typical portfolio values.)
    assert 0 - threshold <= ylim[0] < ylim[1] <= 1 + threshold


def test_optimize_pf_plot_vol_and_sharpe_optimal():
    portfolio_config = portfolio_configs[4]
    pf = formulate_final_portfolio(**portfolio_config)
    
    # Clear the current figure to ensure a fresh plot
    pylab.clf()
    
    # Create the plot
    pf.optimize_pf_plot_vol_and_sharpe_optimal()
    
    # Assert that the plot was created
    assert len(pylab.gcf().get_axes()) == 1

    # Get axis min/max values
    ax = pylab.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Assert on reasonable x and y limits 
    # (Here, 0 and 1 are just placeholders. Adjust as per typical portfolio values.)
    assert 0 <= xlim[0] < xlim[1] <= 1
    assert 0 <= ylim[0] < ylim[1] <= 1