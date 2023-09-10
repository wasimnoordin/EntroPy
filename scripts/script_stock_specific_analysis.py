# Import necessary libraries and modules
import matplotlib.pyplot as pyplot
import pandas 
import datetime
from src.portfolio_composition import formulate_final_portfolio

# Define the path to the stock data file
stock_info_filepath = "/home/wasim/Desktop/EntroPy/data/MAANG_stock_data.csv"
# Load the stock data into a pandas DataFrame
stock_info = pandas.read_csv(stock_info_filepath, index_col="Date", parse_dates=True)
# Build the portfolio using the loaded stock data
final_portfolio = formulate_final_portfolio(stock_data=stock_info)

# Define a function to extract and display attributes of a specific stock
def display_stock_attributes(stock, stock_name):
    print(f"\n{stock_name} Stock Attributes:\n")
    # Display the first five rows of the stock's price history
    print(stock.asset_price_history.head(5))
    # Display the first five rows of the stock's daily return
    print(stock.calculate_investment_daily_return().head(5))
    # Display the stock's forecasted return
    print(stock.forecast_investment_return)
    # Display the stock's annualized volatility
    print(stock.annualised_investment_volatility)
    # Display the stock's skewness
    print(stock.investment_skew)
    # Display the stock's kurtosis
    print(stock.investment_kurtosis)
    # Print the stock object (this might display a summary or other attributes)
    print(stock)
    # Display other attributes of the stock (implementation might be in the stock object)
    stock.display_attributes()

# Define a function to display stock prices for a given date or date range
def display_stock_by_date(stock_ticker, stock_name, portfolio, date_filter, message):
    print(f"\n{stock_name} stock data {message}:")
    # Filter and display the stock data based on the provided date filter
    print(portfolio.asset_price_history[stock_ticker].loc[date_filter])

# Define the stock ticker and its readable name for display
stock_ticker = "AMZN"
stock_readable_name = "Amazon"

# Extract the specified stock's data from the portfolio
selected_stock = final_portfolio.extract_stock(stock_ticker)
# Display the extracted stock's attributes
display_stock_attributes(selected_stock, stock_readable_name)

# Display the stock data for a specific date
date_specific = str(datetime.datetime(2018, 1, 2))
display_stock_by_date(stock_ticker, stock_readable_name, final_portfolio, date_specific, f"on {date_specific}")

# Display the stock data for dates after a specific date
date_after_specific = final_portfolio.asset_price_history.index > datetime.datetime(2018, 1, 2)
display_stock_by_date(stock_ticker, stock_readable_name, final_portfolio, date_after_specific, "after 2018-01-01")

# Display the stock data for a specific year
year_specific = 2022
date_year_specific = final_portfolio.asset_price_history.index.year == year_specific
display_stock_by_date(stock_ticker, stock_readable_name, final_portfolio, date_year_specific, f"for the year {year_specific}")
