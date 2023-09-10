import datetime
import matplotlib.pyplot as pyplot
import pandas
from src.portfolio_composition import formulate_final_portfolio

# Set the style for the plots
pyplot.style.use("bmh")

# Configure plot line settings
pyplot.rcParams["lines.linewidth"] = 2  
pyplot.rcParams["lines.color"] = "black"

# Configure grid settings for the plot
pyplot.rcParams["grid.alpha"] = 0.5 

# Set the default figure size for the plots
pyplot.rcParams["figure.figsize"] = (12, 8)  

# Configure x and y tick label sizes
pyplot.rcParams["xtick.labelsize"] = 12  
pyplot.rcParams["ytick.labelsize"] = 12  

# Configure title and axis label settings
pyplot.rcParams["axes.titlesize"] = 18  
pyplot.rcParams["axes.titleweight"] = "bold"  
pyplot.rcParams["axes.labelsize"] = 14  
pyplot.rcParams["axes.labelweight"] = "bold" 

# Specify the path to the stock data CSV file
price_data_filepath = "/home/wasim/Desktop/EntroPy/data/MAANG_stock_data.csv"

# Read the stock data from the CSV file into a pandas DataFrame
price_data = pandas.read_csv(price_data_filepath, index_col="Date", parse_dates=True)

# Create a portfolio object using the stock data
final_portfolio = formulate_final_portfolio(stock_data=price_data)

# Calculate the optimized portfolio weights and results using Monte Carlo simulation
opt_w, opt_res = final_portfolio.pf_mcs_optimised_portfolio(mcs_iterations=5000)

# Visualize the results of the Monte Carlo simulation
final_portfolio.pf_mcs_visualisation()

# Plot the optimized portfolio on the efficient frontier
final_portfolio.optimize_pf_plot_mef()

# Plot the efficient frontier with the Sharpe optimal portfolio
final_portfolio.efficient_frontier_instance.mef_plot_vol_and_sharpe_optimal()

# Visualize the stock data in the portfolio
final_portfolio.pf_stock_visualisation()

# Set the title for the plot
pyplot.title("Overlay of Monte Carlo and Efficient Frontier Results")

# Display the legend in the best location
pyplot.legend(loc='best')

# Display the plot
pyplot.gcf().set_size_inches(12, 8)
pyplot.show()
