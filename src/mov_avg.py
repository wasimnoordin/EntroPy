import numpy
import pandas
import matplotlib.pyplot as pyplot
from typing import Callable
from typing import List

import numpy
from typing import Callable, List
import pandas
from matplotlib import pyplot

# Define the main function for calculating moving averages
def moving_average_calculator(input_stock_prices, moving_avg_function: Callable, window_sizes: List[int], visualise: bool=True):
    """
    Main function to calculate moving averages, determine buy/sell signals, and optionally visualize the results.
    """
    
    # Validate and preprocess the input stock prices
    input_stock_prices = validate_and_preprocess_input(input_stock_prices)
    
    # Compute moving averages for the given window sizes
    moving_avg_prices = compute_moving_averages(input_stock_prices, moving_avg_function, window_sizes)
    
    # Determine buy and sell signals based on moving average crossovers
    bs_signals = determine_buy_sell_signals(moving_avg_prices, min(window_sizes), max(window_sizes))
    
    # Visualize the moving averages and buy/sell signals if the 'visualise' flag is set to True
    if visualise:
        visualize_data(moving_avg_prices, bs_signals, moving_avg_function, min(window_sizes), max(window_sizes))
    
    return moving_avg_prices

def validate_and_preprocess_input(input_stock_prices):
    """
    Validates the input stock prices and preprocesses it if necessary.
    """
    
    # Check if the input is either a pandas Series or DataFrame
    if not isinstance(input_stock_prices, (pandas.Series, pandas.DataFrame)):
        raise ValueError("Prices required as type pandas.Series or pandas.DataFrame.")
    
    # Convert Series to DataFrame for uniformity
    if isinstance(input_stock_prices, pandas.Series):
        input_stock_prices = input_stock_prices.to_frame()
    return input_stock_prices

def compute_moving_averages(data, moving_avg_function: Callable, window_sizes: List[int]) -> pandas.DataFrame:
    """
    Computes moving averages for the given window sizes using the specified moving average function.
    """
    
    # Create a deep copy of the input data to avoid modifying the original
    moving_avg_prices = data.copy(deep=True)
    
    # Calculate moving averages for each window size and add as new columns to the DataFrame
    for window in window_sizes:
        indicator = f"{window}d"
        moving_avg_prices[indicator] = moving_avg_function(data, window_size=window).iloc[:, 0]
    return moving_avg_prices

def determine_buy_sell_signals(moving_avg_prices, min_window_size, max_window_size):
    """
    Determines buy/sell signals based on crossovers between short-term and long-term moving averages.
    """
    
    # Create a deep copy of the moving average prices to avoid modifying the original
    bs_signals = moving_avg_prices.copy(deep=True)
    
    # Calculate deviation based on crossover of short-term and long-term moving averages
    bs_signals['deviation'] = numpy.where(moving_avg_prices[f"{min_window_size}d"] > moving_avg_prices[f"{max_window_size}d"], 1.0, 0.0)
    
    # Determine buy/sell signals based on changes in the deviation
    bs_signals['signal'] = bs_signals['deviation'].diff()
    return bs_signals

def visualize_data(ma_data, bs_sig, moving_avg_function: Callable, min_window, max_window):
    """
    Visualizes moving averages, buy signals, and sell signals on a plot.
    """
    
    # Initialize a plot with specified size
    fig, ax = pyplot.subplots(figsize=(12, 8))
    
    # Plot the moving averages on the graph
    ma_data.plot(ax=ax)
    
    # Extract indices where buy and sell signals occur
    buy_indices = bs_sig[bs_sig['signal'] == 1.0].index
    sell_indices = bs_sig[bs_sig['signal'] == -1.0].index
    
    # Plot buy signals (green) and sell signals (red) on the graph
    ax.plot(buy_indices, ma_data[f"{min_window}d"][buy_indices], 'P', markersize=15, color='green', label='buy signal')
    ax.plot(buy_indices, ma_data[f"{min_window}d"][buy_indices], color='green', linestyle='-')
    ax.plot(sell_indices, ma_data[f"{min_window}d"][sell_indices], 'X', markersize=15, color='red', label='sell signal')
    ax.plot(sell_indices, ma_data[f"{min_window}d"][sell_indices], color='red', linestyle='-')
    
    # Set title, labels, and legend for the plot
    ax.set_title(f"Band of Moving Averages ({moving_avg_function.__name__})")
    ax.set_xlabel(ma_data.index.name)
    ax.set_ylabel("Price")
    ax.legend(loc='best')
    pyplot.tight_layout()

def simple_moving_average_mean(
        input_stock_prices: pandas.DataFrame,
        window_size: int = 50,
        min_periods: int = None,
        window_index: bool = False,
        win_type: str = None,
        on: str = None,
        axis: int = 0,
        closed: str = None) -> pandas.DataFrame:
    """
    Calculates the simple moving average for the given input stock prices.
    """
    
    # Validate the input parameters
    if not isinstance(input_stock_prices, pandas.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if window_size <= 0:
        raise ValueError("Window must be greater than 0.")
    if min_periods is not None and min_periods < 0:
        raise ValueError("Min_periods must be non-negative.")
    if win_type is not None and not isinstance(win_type, str):
        raise ValueError("Win_type must be a string.")
    if on is not None and not isinstance(on, str):
        raise ValueError("On must be a string.")
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1.")
    if closed is not None and closed not in ['RHS', 'LHS', 'both', 'neither']:
        raise ValueError("Closed must be one of 'RHS', 'LHS', 'both', or 'neither'.")
    if input_stock_prices.empty:
        raise ValueError("Input data must not be empty.")

    # Create a rolling window object for the input stock prices
    simple_moving_window = input_stock_prices.rolling(
        window=window_size, 
        min_periods=min_periods, 
        center=window_index, 
        win_type=win_type, 
        on=on, 
        axis=axis, 
        closed=closed)

    # Compute the mean for each window to get the simple moving average
    simple_moving_average = simple_moving_window.mean()

    return simple_moving_average
import pandas

def exponential_moving_average_mean(
        input_stock_prices: pandas.DataFrame,
        window_size: int = 50,
        min_periods: int = None,
        com: float = None,
        halflife: float = None,
        alpha: float = None,
        decay_adj_factor: bool = False,
        ignore_na: bool = False,
        axis: int = 0,
        times: str = None,
        method: str = 'single') -> pandas.DataFrame:
    """
    Calculates the exponential moving average for the given input stock prices.
    """
    
    # Validate the input parameters
    if not isinstance(input_stock_prices, pandas.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if window_size <= 0:
        raise ValueError("Window must be greater than 0.")
    if com is not None and com < 0:
        raise ValueError("Com must be non-negative.")
    if halflife is not None and halflife <= 0:
        raise ValueError("Halflife must be greater than 0.")
    if alpha is not None and (alpha <= 0 or alpha >= 1):
        raise ValueError("Alpha must be between 0 and 1 exclusive.")
    if min_periods is not None and min_periods < 0:
        raise ValueError("Min_periods must be non-negative.")
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1.")
    if method not in ['single', 'double', 'triple']:
        raise ValueError("Method must be one of 'single', 'double', or 'triple'.")
    if input_stock_prices.empty:
        raise ValueError("Input data must not be empty.")
    
    # Create an exponential moving window object for the input stock prices
    exponential_moving_window = input_stock_prices.ewm(
        span=window_size,
        com=com,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods if min_periods is not None else window_size,
        adjust=decay_adj_factor,
        ignore_na=ignore_na,
        axis=axis,
        times=times,
        method=method
    )

    # Compute the mean for the exponential moving window
    exponential_moving_average = exponential_moving_window.mean()

    return exponential_moving_average

def simple_moving_average_standard_deviation(
        input_stock_prices: pandas.DataFrame,
        window_size: int = 50,
        min_periods: int = None,
        center: bool = False,
        win_type: str = None,
        on: str = None,
        axis: int = 0,
        closed: str = None,
        ddof: int = 1) -> pandas.DataFrame:
    """
    Calculates the standard deviation for a simple moving average of the given input stock prices.
    """
    
    # Validate the input parameters
    if not isinstance(input_stock_prices, pandas.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if window_size <= 0:
        raise ValueError("Window must be greater than 0.")
    if min_periods is not None and min_periods < 0:
        raise ValueError("Min_periods must be non-negative.")
    if win_type is not None and not isinstance(win_type, str):
        raise ValueError("Win_type must be a string.")
    if on is not None and not isinstance(on, str):
        raise ValueError("On must be a string.")
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1.")
    if closed is not None and closed not in ['RHS', 'LHS', 'both', 'neither']:
        raise ValueError("Closed must be one of 'RHS', 'LHS', 'both', or 'neither'.")
    if ddof < 0:
        raise ValueError("Degrees of freedom (ddof) must be non-negative.")
    if input_stock_prices.empty:
        raise ValueError("Input data must not be empty.")
    
    # Apply the rolling window to the input data
    simple_ma_rolling_window = input_stock_prices.rolling(
        window=window_size,
        min_periods=min_periods,
        center=center,
        win_type=win_type,
        on=on,
        axis=axis,
        closed=closed
    )

    # Compute the standard deviation for each window in the rolling window
    simple_ma_standard_deviation = simple_ma_rolling_window.std(ddof=ddof)

    return simple_ma_standard_deviation

def exponential_moving_average_standard_deviation(
        input_stock_prices: pandas.DataFrame,
        window_size: int = 50,
        com: float = None,
        halflife: float = None,
        alpha: float = None,
        min_periods: int = None,
        adjust: bool = False,
        ignore_na: bool = False,
        axis: int = 0,
        times: str = None,
        method: str = 'single') -> pandas.DataFrame:
    """
    Calculates the standard deviation for an exponential moving average of the given input stock prices.
    """
    
    # Validate the input parameters
    if not isinstance(input_stock_prices, pandas.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if window_size <= 0:
        raise ValueError("Window must be greater than 0.")
    if com is not None and com < 0:
        raise ValueError("Com must be non-negative.")
    if halflife is not None and halflife <= 0:
        raise ValueError("Halflife must be greater than 0.")
    if alpha is not None and (alpha <= 0 or alpha >= 1):
        raise ValueError("Alpha must be between 0 and 1 exclusive.")
    if min_periods is not None and min_periods < 0:
        raise ValueError("Min_periods must be non-negative.")
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1.")
    if method not in ['single', 'double', 'triple']:
        raise ValueError("Method must be one of 'single', 'double', or 'triple'.")
    if input_stock_prices.empty:
        raise ValueError("Input data must not be empty.")
    
    # Create an exponential moving window object for the input stock prices
    exponential_moving_window = input_stock_prices.ewm(
        span=window_size,
        com=com,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods if min_periods is not None else window_size,
        adjust=adjust,
        ignore_na=ignore_na,
        axis=axis,
        times=times,
        method=method
    )

    # Compute the standard deviation for the exponential moving window
    exponential_ma_standard_deviation = exponential_moving_window.std()

    return exponential_ma_standard_deviation
