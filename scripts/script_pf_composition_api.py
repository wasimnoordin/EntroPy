import datetime
import pandas as pd

# Import the necessary function for portfolio formulation
from src.portfolio_composition import formulate_final_portfolio

# Define the allocation percentages for each stock in the portfolio
stock_allocation = {
    "META": 20,
    "AAPL": 30,
    "AMZN": 25,
    "NFLX": 15,
    "GOOG": 10,
}

# Convert the stock allocation dictionary into a pandas DataFrame for easier manipulation
allocation_df = pd.DataFrame(list(stock_allocation.items()), columns=["Name", "Allocation"])

# Print out the stock allocation to the console for user visibility
print("\n" + "="*40)
print("STOCK ALLOCATION".center(40))
print("="*40 + "\n")
print(allocation_df)

# Define the time range for which we want to retrieve stock data
start_date = datetime.datetime(2018, 1, 1)
end_date = "2023-1-1"

# Specify the financial index against which the portfolio performance will be compared
financial_index = "^GSPC"

# Use the provided function to construct the portfolio based on the defined stock allocation and date range
portfolio = formulate_final_portfolio(
    stock_symbols=allocation_df["Name"].tolist(),
    apportionment=allocation_df,
    start=start_date,
    end=end_date,
    api_type="yfinance",  # Specify the data source as 'yfinance'
    financial_index=financial_index,
)

# Display the distribution of stocks in the portfolio
print("\n" + "="*40)
print("PORTFOLIO DISTRIBUTION".center(40))
print("="*40 + "\n")
print(portfolio.portfolio_distribution)

# Display the historical price data of the assets in the portfolio
print("\n" + "="*40)
print("ASSET PRICE HISTORY".center(40))
print("="*40 + "\n")
print(portfolio.asset_price_history.head(5))  # Displaying only the first 5 rows for brevity

# Print a summary of the portfolio's attributes and characteristics
print("\n" + "="*40)
print("PORTFOLIO SUMMARY".center(40))
print("="*40 + "\n")
print(portfolio)
portfolio.pf_print_portfolio_attributes()
