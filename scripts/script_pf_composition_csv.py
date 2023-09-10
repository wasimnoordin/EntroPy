import datetime
import pandas

# Define file paths for portfolio and stock data
PORTFOLIO_FILE = "/home/wasim/Desktop/EntroPy/data/MAANG_portfolio.csv"
STOCKDATA_FILE = "/home/wasim/Desktop/EntroPy/data/MAANG_stock_data.csv"

# Import the necessary function for portfolio formulation from the specified module
from src.portfolio_composition import formulate_final_portfolio

# Load the portfolio allocation and stock price data from the specified file paths
allocation_data_filepath = PORTFOLIO_FILE
price_data_filepath = STOCKDATA_FILE

allocation_data = pandas.read_csv(allocation_data_filepath)
price_data = pandas.read_csv(price_data_filepath, index_col="Date", parse_dates=True)

# Define a helper function to display data sections with a formatted title
def display_section(title, data):
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")
    print(data, "\n")

# Display an overview of the loaded portfolio allocation and the first few rows of the stock price data
display_section("LOADED PORTFOLIO DATA", allocation_data)
display_section("FIRST ROWS OF STOCK DATA", price_data.head())

# Construct a portfolio based on the stock price data with equal weights for each stock
equal_weight_portfolio = formulate_final_portfolio(stock_data=price_data)

# Display the distribution of stocks in the equal-weighted portfolio and its price history
display_section("PORTFOLIO WITH EQUAL WEIGHTS", equal_weight_portfolio.portfolio_distribution)
display_section("PRICE HISTORY (EQUAL WEIGHTS)", equal_weight_portfolio.asset_price_history.head())

# Construct a portfolio based on the stock price data and the specified allocation data
custom_weight_portfolio = formulate_final_portfolio(stock_data=price_data, apportionment=allocation_data)

# Display the distribution of stocks in the custom-weighted portfolio and its price history
display_section("PORTFOLIO WITH CUSTOM WEIGHTS", custom_weight_portfolio.portfolio_distribution)
display_section("PRICE HISTORY (CUSTOM WEIGHTS)", custom_weight_portfolio.asset_price_history.head())
