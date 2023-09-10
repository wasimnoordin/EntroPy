import numpy
import pandas 
import pytest
import yfinance
from src.index import Index
from src.portfolio_composition import formulate_final_portfolio

def create_stock_app_dict():
    """
    Generate a stock allocation dictionary.
    """
    apportionment = {
        "META": 20,
        "AAPL": 30,
        "AMZN": 25,
        "NFLX": 15,
        "GOOG": 10
    }
    return apportionment

def convert_allocation_to_dataframe(allocation_dict):
    """
    Convert the stock allocation dictionary to a pandas DataFrame.
    """
    index_dataframe = pandas.DataFrame(list(allocation_dict.items()), columns=["Name", "Allocation"])
    return index_dataframe

def extract_stock_symbols(index_dataframe):
    """
    Extract stock symbols from the allocation DataFrame.
    """
    return index_dataframe["Name"].values.tolist()

# Create a stock allocation dictionary and convert it to a DataFrame
stock_apportionment = create_stock_app_dict()
apportionment = convert_allocation_to_dataframe(stock_apportionment)
yf_stock_symbols = extract_stock_symbols(apportionment)

# Define the trading horizon
start = "2020-01-01"
end = "2023-01-01"

def test_index():
    # Create a portfolio using the formulated final portfolio function
    pf = formulate_final_portfolio(
        stock_symbols=yf_stock_symbols,
        apportionment=apportionment,
        start=start,
        end=end,
        api_type="yfinance",
        financial_index="^GSPC",
    )
    # Assert checks for the portfolio's beta coefficient and financial index attributes
    assert pf.beta_coefficient is not None
    assert pf.financial_index.investment_name == "^GSPC"
    assert isinstance(pf.financial_index, Index)

def test_index_initialization():
    # Sample asset price history for the index
    asset_price_history = pandas.Series([100, 102, 101, 103], name="^GSPC")
    
    # Initialize the Index
    index = Index(asset_price_history)
    
    # Assert checks to ensure the index object is correctly initialized
    assert isinstance(index, Index)
    assert index.asset_price_history.equals(asset_price_history)

def test_index_daily_returns():
    # Sample asset price history for the index
    asset_price_history = pandas.Series([100, 102, 101, 103], name="^GSPC")

    # Initialize the Index
    index = Index(asset_price_history)

    # Define expected daily returns based on the sample data
    expected_returns = pandas.Series([0.02, -0.00980392, 0.01980198], index=[1,2,3], name="^GSPC")

    # Drop NaN values from the computed returns
    computed_returns = index.calculate_daily_return.dropna()

    # Print both the expected and computed returns for comparison
    print("Adjusted Expected returns:\n", expected_returns)
    print("Computed returns:\n", computed_returns)

    # Check if the computed daily returns match the expected returns
    assert numpy.allclose(computed_returns, expected_returns, atol=1e-8)

def test_index_name_and_category():
    # Sample asset price history for the index
    asset_price_history = pandas.Series([100, 102, 101, 103], name="^GSPC")
    
    # Initialize the Index
    index = Index(asset_price_history)
    
    # Assert checks to ensure the index's name and category attributes are correctly set
    assert index.investment_name == "^GSPC"
    assert index.investment_category == "Financial Index"
