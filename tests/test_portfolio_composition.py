import pandas

import datetime
import pytest

import matplotlib.pylab as plt
import yfinance

from src.portfolio_optimisation import Portfolio_Optimised_Functions
from src.portfolio_composition import formulate_final_portfolio
from src.stock import Stock

plt.switch_backend("Agg")

# define paths for data from file
allocation_data_file = "/home/wasim/Desktop/EntroPy/data/MAANG_portfolio.csv"
price_data_file = "/home/wasim/Desktop/EntroPy/data/MAANG_stock_data.csv"

# portfolio allocation (yfinance):
yf_apportionment_file = pandas.read_csv(allocation_data_file)
yf_apportionment = yf_apportionment_file.copy()

# stock price data (yfinance):
yf_price_data_file = pandas.read_csv(price_data_file, index_col="Date", parse_dates=True)

# create testing variables
names_yf = yf_apportionment.Name.values.tolist()

# weights
equal_allocations = [1.0 / len(names_yf) for i in range(len(names_yf))]
yf_allocations = pandas.DataFrame({"Allocation": equal_allocations, "Name": names_yf})
start = datetime.datetime(2018, 1, 1)
end = "2023-01-01"

# create kwargs to be passed to formulate_final_portfolio function
# Base configuration
base_config = {
    "stock_symbols": names_yf,
    "api_type": "yfinance"
}

# Function to create a new config based on the base configuration
def create_config(**overrides):
    config = base_config.copy()
    config.update(overrides)
    return config

portfolio_configs = [
    create_config(apportionment=yf_apportionment),
    create_config(),  # No changes from base config
    create_config(start=start, end=end),
    create_config(start=start, end=end),  # api_type is already in base config
    {"stock_data": yf_price_data_file},   # Completely different from base, so define separately
    {"stock_data": yf_price_data_file, "apportionment": yf_apportionment_file}  # Same as above
]

# Error limts
max_error = 1e-12
min_error = 1e-6

# --------------- PORTFOLIO CONSTRUCTION TESTS ---------------

def test_portfolio_construction_with_apportionment():
    """
    Test the construction of a portfolio using a configuration with apportionment data.
    Validates various properties of the portfolio, including the allocation of assets.
    """

    # Retrieve the specific portfolio configuration
    config = portfolio_configs[0]
    # Create the portfolio instance using the given configuration
    pf = formulate_final_portfolio(**config)
    
    # Check if the constructed portfolio and stock instances are of the correct type
    assert isinstance(pf, Portfolio_Optimised_Functions)
    assert isinstance(pf.extract_stock(names_yf[0]), Stock)
    # Check if the portfolio's asset prices and distribution are of type DataFrame
    assert isinstance(pf.asset_price_history, pandas.DataFrame)
    assert isinstance(pf.portfolio_distribution, pandas.DataFrame)
    
    # Assert that the number of stocks matches the number of columns in the asset price history
    assert len(pf.stock_objects) == len(pf.asset_price_history.columns)
    # Assert that the columns in the asset price history match the stock names
    assert pf.asset_price_history.columns.tolist() == names_yf
    # Assert that the index name for the asset price history is "Date"
    assert pf.asset_price_history.index.name == "Date"
    # Assert that the portfolio's distribution matches the predefined apportionment
    assert (pf.portfolio_distribution == yf_apportionment).all().all()
    
    # Calculate the proportioned allocation
    allocations = pf.calculate_pf_proportioned_allocation()
    # Ensure each weight is between 0 and 1
    for allocation in allocations:
        assert 0 <= allocation <= 1
    # Ensure the sum of all weights is approximately 1
    assert abs(sum(allocations) - 1) <= 1e-9  # Small tolerance for potential floating-point inaccuracies

def test_portfolio_construction_base_config():
    """
    Test the construction of a portfolio using the base configuration.
    Validates the properties of the portfolio such as asset price history, stock object count, distribution, and proportioned allocation.
    """

    # Retrieve the specific portfolio configuration from the predefined list
    config = portfolio_configs[1]
    
    # Create the final portfolio instance using the given configuration
    final_portfolio = formulate_final_portfolio(**config)

    # Gather data from the portfolio for assertions
    portfolio_asset_prices = final_portfolio.asset_price_history
    portfolio_distribution = final_portfolio.portfolio_distribution
    portfolio_num_stocks = len(portfolio_asset_prices.columns)
    
    # Make assertions to ensure correctness of the portfolio and its attributes
    
    # Assert that the portfolio and stock instances are of the correct type
    assert isinstance(final_portfolio, Portfolio_Optimised_Functions)
    assert isinstance(final_portfolio.extract_stock(names_yf[0]), Stock)
    
    # Assert that the portfolio's asset prices and distribution are dataframes
    assert isinstance(portfolio_asset_prices, pandas.DataFrame)
    assert isinstance(portfolio_distribution, pandas.DataFrame)
    
    # Assert that the number of stocks matches the number of columns in the asset price history
    assert portfolio_num_stocks == len(names_yf)
    # Assert that the columns in the asset price history match the stock names
    assert portfolio_asset_prices.columns.tolist() == names_yf
    
    # Assert that the index name for the asset price history is "Date"
    assert portfolio_asset_prices.index.name == "Date"
    # Assert that the portfolio's distribution matches the predefined allocations
    assert (portfolio_distribution == yf_allocations).all().all()
    # Assert that the portfolio's proportioned allocation is close to equal allocations
    assert (final_portfolio.calculate_pf_proportioned_allocation() - equal_allocations <= max_error).all()
    
    # Print portfolio attributes for further inspection (useful during development/testing)
    final_portfolio.pf_print_portfolio_attributes()

def test_portfolio_construction_with_date_range():
    """
    Test the construction of a portfolio using a configuration with a specified date range.
    Validates the properties of the portfolio such as asset price history, stock object count, distribution, and proportioned allocation.
    """
        
    # Retrieve the specific portfolio configuration
    config = portfolio_configs[3]
    # Create the final portfolio instance using the given configuration
    final_portfolio = formulate_final_portfolio(**config)
    
    # Check if the constructed portfolio and stock instances are of the correct type
    assert isinstance(final_portfolio, Portfolio_Optimised_Functions)
    assert isinstance(final_portfolio.extract_stock(names_yf[0]), Stock)
    # Check if the portfolio's asset prices and distribution are of type DataFrame
    assert isinstance(final_portfolio.asset_price_history, pandas.DataFrame)
    assert isinstance(final_portfolio.portfolio_distribution, pandas.DataFrame)
    
    # Assert that the number of stocks matches the number of columns in the asset price history
    assert len(final_portfolio.stock_objects) == len(final_portfolio.asset_price_history.columns)
    # Assert that the columns in the asset price history match the stock names
    assert final_portfolio.asset_price_history.columns.tolist() == names_yf
    # Assert that the index name for the asset price history is "Date"
    assert final_portfolio.asset_price_history.index.name == "Date"
    # Assert that the portfolio's distribution matches the predefined allocations
    assert (final_portfolio.portfolio_distribution == yf_allocations).all().all()
    # Assert that the portfolio's proportioned allocation is close to equal allocations
    assert (final_portfolio.calculate_pf_proportioned_allocation() - equal_allocations <= max_error).all()
    
     # Print portfolio attributes for further inspection
    final_portfolio.pf_print_portfolio_attributes()

# --------------- PORTFOLIO VISUALISATION TESTS ---------------

def test_pf_stock_visualisation():
    portfolio_config = portfolio_configs[4]
    pf = formulate_final_portfolio(**portfolio_config)
    
    # Clear the current figure to ensure a fresh plot
    plt.clf()
    
    # Create the plot
    pf.pf_stock_visualisation()
    
    # Assert that the plot was created
    assert len(plt.gcf().get_axes()) == 1

    # Get axis min/max values
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Assert on reasonable x and y limits 
    # (Here, 0 and 1 are just placeholders. Adjust as per typical stock values.)
    assert 0 <= xlim[0] < xlim[1] <= 1
    assert 0 <= ylim[0] < ylim[1] <= 1
