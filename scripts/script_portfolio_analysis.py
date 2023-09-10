import matplotlib.pyplot as pyplot
import pandas 
import datetime
from src.portfolio_composition import formulate_final_portfolio

# Set the style for the plots
pyplot.style.use("bmh")

# Adjust various properties for the plots
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

# Load the stock data from the specified file path
stock_info_filepath = "/home/wasim/Desktop/EntroPy/data/MAANG_stock_data.csv"
stock_info = pandas.read_csv(stock_info_filepath, index_col="Date", parse_dates=True)

# Construct the portfolio using the loaded stock data
final_portfolio = formulate_final_portfolio(stock_data=stock_info)

# Plot the asset price history of the portfolio
ax1 = final_portfolio.asset_price_history.plot()
ax1.set_title("Portfolio Asset Price History")
pyplot.show()

# Calculate and plot the cumulative returns of the portfolio
cumulative_returns = final_portfolio.calculate_pf_cumulative_return()
ax2 = cumulative_returns.plot()
ax2.set_title("Portfolio Cumulative Returns")
ax2.axhline(y=0, color="black", lw=1.0)  # Add a horizontal line at y=0 for reference
pyplot.show()

# Calculate and plot the daily percentage changes of the portfolio returns
daily_percentage_changes = final_portfolio.calculate_pf_daily_return()
ax3 = daily_percentage_changes.plot()
ax3.set_title("Daily Percentage Changes of Returns")
ax3.axhline(y=0, color="black", lw=1.0)  # Add a horizontal line at y=0 for reference
pyplot.show()

# Calculate and plot the daily logarithmic returns of the portfolio
daily_log_returns = final_portfolio.calculate_pf_daily_return_logarithmic()
ax4 = daily_log_returns.plot()
ax4.set_title("Daily Log Returns")
ax4.axhline(y=0, color="black", lw=1.0)  # Add a horizontal line at y=0 for reference
pyplot.show()

# Calculate and plot the cumulative logarithmic returns of the portfolio
cumulative_log_returns = final_portfolio.calculate_pf_daily_return_logarithmic().cumsum()
ax5 = cumulative_log_returns.plot()
ax5.set_title("Cumulative Log Returns")
ax5.axhline(y=0, color="black", lw=1.0)  # Add a horizontal line at y=0 for reference
pyplot.show()
