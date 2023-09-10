import matplotlib.pyplot as pyplot
import pandas 
import datetime
from src.portfolio_composition import formulate_final_portfolio
from src.bbands import bollinger_bands

from src.mov_avg import (
    simple_moving_average_mean,
    exponential_moving_average_mean,
)

# Style settings for plots
pyplot.style.use("bmh")
# Adjust the properties:
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

# Load data and build portfolio
stock_info_filepath = "/home/wasim/Desktop/EntroPy/data/MAANG_stock_data.csv"
stock_info = pandas.read_csv(stock_info_filepath, index_col="Date", parse_dates=True)
final_portfolio = formulate_final_portfolio(stock_data=stock_info)

# Extracting stock data
stock_symbol = "AAPL"
stock_data = final_portfolio.extract_stock(stock_symbol).asset_price_history.copy(deep=True)
stock_name = stock_symbol.split("/")[-1]  # Extracting the stock's name from the symbol

span = 20 # Medium-term stategy that uses 2σ (short-term: 10d and 1.5σ, long-term: 50d and 2.5σ)

# Plot Bollinger Band (simple moving average) for the given stock
bollinger_bands(stock_data, simple_moving_average_mean, span)
# Set title after calling bollinger_bands, adapting to the stock's name
pyplot.title(f"Bollinger Band \u00B12σ (SMA) for {stock_name}")
pyplot.show()

# Plot Bollinger Band (exponential moving average) for the given stock
bollinger_bands(stock_data, exponential_moving_average_mean, span)
# Set title after calling bollinger_bands, adapting to the stock's name
pyplot.title(f"Bollinger Band \u00B12σ (EMA) for {stock_name}")
pyplot.show()