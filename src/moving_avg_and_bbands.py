import numpy
import pandas
import matplotlib.pyplot as pyplot
from typing import Callable
from typing import List

# Define the main function for calculating moving averages
def moving_average_calculator(input_stock_prices: pandas.DataFrame, moving_avg_function: Callable, window_sizes: List[int], visualise: bool = True) -> pandas.DataFrame:
    # Prepare data for moving average calculation
    moving_average_values = _prepare_ma_calculator_data(input_stock_prices)
    
    # Compute moving averages for various window sizes
    _compute_ma_calculator_averages(moving_average_values, moving_avg_function, window_sizes)
    
    # Visualize if required
    if visualise:
        _plot_moving_averages(moving_average_values, moving_avg_function, window_sizes)
    
    # Return the DataFrame containing moving average values
    return moving_average_values

# Prepare input stock price data for calculation
def _prepare_ma_calculator_data(input_stock_prices):
    if not isinstance(input_stock_prices, (pandas.Series, pandas.DataFrame)):
        raise ValueError("data is expected to be of type pandas.Series or pandas.DataFrame")
    return input_stock_prices.to_frame() if isinstance(input_stock_prices, pandas.Series) else input_stock_prices.copy(deep=True)

# Compute moving averages for different window sizes
def _compute_ma_calculator_averages(moving_average_values, moving_avg_function, window_sizes):
    # Iterate through the window sizes and apply the moving average function
    for window_size in window_sizes:
        column_name = str(window_size) + "d"
        moving_average = moving_avg_function(moving_average_values, span=window_size)
        moving_average_values[column_name] = moving_average

# Plot moving averages and associated signals
def _plot_moving_averages(moving_average_values, moving_average_function, window_sizes):
    figure, axes = pyplot.subplots(figsize=(10, 6))
    moving_average_values.plot(axes=axes, linewidth=2)
    _create_and_plot_signals(axes, moving_average_values, window_sizes)
    pyplot.title(f"Moving Average Envelope ({moving_average_function.__name__})", fontsize=14)
    pyplot.legend(ncol=2, fontsize=10)
    pyplot.xlabel(moving_average_values.index.name, fontsize=12)
    pyplot.ylabel("Value", fontsize=12)
    pyplot.grid(True, linestyle='--')

# Create and plot buy and sell signals on the moving average plot
def _create_and_plot_signals(axes, moving_average_values, window_sizes):
    minimum_window_size, maximum_window_size = min(window_sizes), max(window_sizes)
    identifier = f"{minimum_window_size}d"
    signals = _compute_crossover_signals(moving_average_values, minimum_window_size, maximum_window_size)

    # Extract buy and sell signals
    buy_sign = signals[signals["crossover"] == 1.0]
    sell_sign = signals[signals["crossover"] == -1.0]

    # Customize marker styles for buy and sell signals
    buy_marker_style = {"marker": "8", "s": 150, "color": "r", "label": "buy signal", "edgecolors": "black"}
    sell_marker_style = {"marker": "*", "s": 150, "color": "m", "label": "sell signal", "edgecolors": "black"}

    # Plot buy and sell signals on the moving average plot
    _plot_signals(axes, buy_sign, identifier, buy_marker_style)
    _plot_signals(axes, sell_sign, identifier, sell_marker_style)

# Plot buy and sell signals on the moving average plot
def _plot_signals(axes, signals, identifier, marker_style):
    axes.scatter(signals.index.values, signals[identifier].values, **marker_style)

# Compute crossover signals between moving averages
def _compute_crossover_signals(moving_average_values, minimum_window_size, maximum_window_size):
    minlabel = f"{minimum_window_size}d"
    maxlabel = f"{maximum_window_size}d"
    signals = moving_average_values.copy()
    signals["crossover"] = (moving_average_values[minlabel][minimum_window_size:] > moving_average_values[maxlabel][minimum_window_size:]).astype(float) * 2 - 1
    signals["crossover"] = signals["crossover"].diff()
    return signals

"___________"

