import pandas
import datetime
import yfinance
import pytest

from src.stock import Stock
from src.portfolio_composition import formulate_final_portfolio

# Class to set up the test environment for Stock tests
class StockTestSetup:
    # Define error tolerances
    lower_tol = 1e-8
    upper_tol = 1e-20
    # Define file paths for portfolio and stock data
    portfolio_file = "/home/wasim/Desktop/EntroPy/data/MAANG_portfolio.csv"
    stock_prices_file = "/home/wasim/Desktop/EntroPy/data/MAANG_stock_data.csv"

    def __init__(self):
        # Load portfolio and stock data
        self.df_pf = self._load_portfolio_data()
        self.df_data = self._load_stock_data()
        # Get stock names
        self.names_yf = self._get_names()
        # Define date range for the tests
        self.start = datetime.datetime(2018, 1, 1)
        self.end = "2023-01-01"
        # Build the dictionary to pass to the tests
        self.d_pass = self._build_d_pass()

    # Load portfolio data from CSV
    def _load_portfolio_data(self):
        return pandas.read_csv(self.portfolio_file)

    # Load stock data from CSV
    def _load_stock_data(self):
        return pandas.read_csv(self.stock_prices_file, index_col="Date", parse_dates=True)

    # Extract stock names from the portfolio data
    def _get_names(self):
        return self.df_pf.Name.values.tolist()

    # Build a dictionary with necessary parameters for the tests
    def _build_d_pass(self):
        return [
            {
                "stock_symbols": self.names_yf,
                "start": self.start,
                "end": self.end,
                "api_type": "yfinance",
            }
        ]

# Initialize the setup at the beginning of the tests
setup = StockTestSetup()

# Test the Stock class functionality
def test_stock():
    d = setup.d_pass[0]
    pf = formulate_final_portfolio(**d)
    # Iterate through each stock object and perform assertions
    for i in range(len(pf.stock_objects)):
        # Check if the extracted stock is an instance of the Stock class
        assert isinstance(pf.extract_stock(setup.names_yf[0]), Stock)
        # Extract the stock and perform further assertions
        stock = pf.extract_stock(setup.names_yf[i])
        # Check if the stock name matches
        assert stock.investment_name == pf.portfolio_distribution["Name"][i]
        # Check if the stock's price history matches with the portfolio's data
        assert all(stock.asset_price_history - pf.asset_price_history[stock.investment_name].to_frame() <= setup.upper_tol)
        # Check if the stock details match with the portfolio's data
        assert all(
            stock.stock_details == pf.portfolio_distribution.loc[pf.portfolio_distribution["Name"] == stock.investment_name]
        )

# Test the initialization of the Stock class
def test_stock_initialization():
    # Define stock details and price history
    stock_details = pandas.DataFrame({"Name": ["AAPL"]})
    asset_price_history = pandas.Series([120, 121, 122, 123], name="AAPL")
    # Initialize the Stock class
    stock = Stock(stock_details, asset_price_history)
    # Check if the stock name was correctly initialized
    assert stock.investment_name.equals(pandas.Series(["AAPL"]))

# Test the calculation of the beta coefficient for a stock
def test_beta_coefficient():
    # Define stock details, price history, and index returns
    stock_details = pandas.DataFrame({"Name": ["AAPL"]})
    asset_price_history = pandas.Series([120, 121, 122, 123], name="AAPL")
    index_returns = pandas.Series([0.01, 0.02, 0.03, 0.008], name="Returns")
    # Initialize the Stock class
    stock = Stock(stock_details, asset_price_history)
    stock_daily_returns = stock.calculate_investment_daily_return()
    # Adjust index returns length if necessary
    if len(stock_daily_returns) != len(index_returns):
        index_returns = index_returns.iloc[:len(stock_daily_returns)]
    # Calculate the beta coefficient
    beta = stock.calculate_beta_coefficient(index_returns)
    # Check if beta was correctly computed
    assert isinstance(beta, float)

# Test the display of stock attributes
def test_display_stock_attributes(capsys):
    # Define stock details and price history
    stock_details = pandas.Series({"Name": "AAPL", "Allocation": "Some Allocation"})
    asset_price_history = pandas.Series([120, 121, 122, 123], name="AAPL")
    # Initialize the Stock class
    stock = Stock(stock_details, asset_price_history)
    # Display the stock attributes
    stock.display_stock_attributes()
    # Capture the printed output
    captured = capsys.readouterr()
    # Check if the output contains the stock name and allocation
    assert "AAPL" in captured.out
    assert "Some Allocation" in captured.out
