import pandas as pd
import yfinance as yf

# Define the portfolio composition
portfolio_data = {
    "Name": ["AAPL", "MSFT", "GOOGL"],
    "Allocation": [0.4, 0.3, 0.3]
}

# Create a DataFrame and save it to CSV
portfolio_df = pd.DataFrame(portfolio_data)
portfolio_csv_path = "data/ex1-portfolio.csv"
portfolio_df.to_csv(portfolio_csv_path, index=False)

# Define the tickers for the assets in the portfolio
tickers = portfolio_data["Name"]

# Fetch the historical price data and combine it into a single DataFrame
stock_data_df = pd.DataFrame()
for ticker in tickers:
    stock_data = yf.Ticker(ticker).history(period="5y")["Close"]
    stock_data_df[ticker] = stock_data

# Save the combined stock data to CSV
stock_data_csv_path = "data/ex1-stockdata.csv"
stock_data_df.to_csv(stock_data_csv_path)
