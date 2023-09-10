import matplotlib.pyplot as pyplot
import pandas 
import datetime

# Import necessary modules and functions
from src.portfolio_composition import formulate_final_portfolio
from src.mov_avg import simple_moving_average_mean, exponential_moving_average_mean, moving_average_calculator
from typing import Callable, List

# Set the style for the plots
pyplot.style.use("bmh")

# Configure the properties for the plots
pyplot.rcParams["lines.linewidth"] = 2  
pyplot.rcParams["lines.color"] = "black"
pyplot.rcParams["grid.alpha"] = 0.5 
pyplot.rcParams["figure.figsize"] = (12, 8)  
pyplot.rcParams["xtick.labelsize"] = 12  
pyplot.rcParams["ytick.labelsize"] = 12  
pyplot.rcParams["axes.titlesize"] = 18  
pyplot.rcParams["axes.titleweight"] = "bold"  
pyplot.rcParams["axes.labelsize"] = 14  
pyplot.rcParams["axes.labelweight"] = "bold"  

# Load the stock data and build the portfolio
stock_info_filepath = "/home/wasim/Desktop/EntroPy/data/MAANG_stock_data.csv"
stock_info = pandas.read_csv(stock_info_filepath, index_col="Date", parse_dates=True)
final_portfolio = formulate_final_portfolio(stock_data=stock_info)

# Extract the time range of the stock data
start_date = final_portfolio.asset_price_history.index.min().strftime('%Y-%m-%d')
end_date = final_portfolio.asset_price_history.index.max().strftime('%Y-%m-%d')

# Plot the simple moving average
ax = final_portfolio.asset_price_history.plot(
    title="Simple Moving Average",
    secondary_y=["AAPL", "GOOG"],
    grid=True,
    label=f"Price Data ({start_date} to {end_date})"
)
simple_moving_average_data = simple_moving_average_mean(
    final_portfolio.asset_price_history,
    window_size=50
)
simple_moving_average_data.plot(
    ax=ax,
    secondary_y=["AAPL", "GOOG"],
    grid=True
)
ax.legend(loc='upper left')
ax.right_ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
pyplot.tight_layout()
pyplot.show()

# Plot the exponential moving average
ax = final_portfolio.asset_price_history.plot(
    title="Exponential Moving Average",
    secondary_y=["AAPL", "GOOG"],
    grid=True,
    label=f"Price Data ({start_date} to {end_date})"
)
exponential_moving_average_data = exponential_moving_average_mean(
    final_portfolio.asset_price_history
)
exponential_moving_average_data.plot(
    ax=ax,
    secondary_y=["AAPL", "GOOG"]
)
ax.legend(loc='upper left')
ax.right_ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
pyplot.tight_layout()
pyplot.show()

# Plot the band of moving averages for a specific stock
stock_symbol = "NFLX"
stock_data = final_portfolio.extract_stock(stock_symbol).asset_price_history.copy(deep=True)

# Convert the stock data to DataFrame if it's a Series
if isinstance(stock_data, pandas.Series):
    stock_data = stock_data.to_frame(name=stock_symbol)

# Define the window sizes for the moving averages
window_sizes = [10, 50, 100, 150, 200]

# Compute and visualize the moving averages
moving_average_calculator(stock_data, exponential_moving_average_mean, window_sizes, visualise=True)

# Set the title for the plot
function_name = exponential_moving_average_mean.__name__.replace("_", " ").title()
title = f"Moving Averages for {stock_symbol} using {function_name}"
pyplot.title(title)
pyplot.legend(loc='best')
pyplot.show()
