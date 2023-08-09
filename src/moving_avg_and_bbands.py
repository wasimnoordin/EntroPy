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
        column_name = str(window_size) + "-day"
        moving_average = moving_avg_function(moving_average_values, window_size=window_size)
        moving_average_values[column_name] = moving_average["Stock"]

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
    min_window_key = "{}d".format(minimum_window_size)
    max_window_key = "{}d".format(maximum_window_size)

    signals = moving_average_values.copy()
    signals["crossover"] = (moving_average_values[min_window_key][minimum_window_size:] > moving_average_values[max_window_key][minimum_window_size:]).astype(float) * 2 - 1
    signals["crossover"] = signals["crossover"].diff()
    return signals

def simple_moving_average_mean(
        input_stock_prices: pandas.DataFrame,
        window_size: int = 50,
        min_periods: int = None,
        window_index: bool = False,
        win_type: str = None,
        on: str = None,
        axis: int = 0,
        closed: str = None) -> pandas.DataFrame:

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
    if closed is not None and closed not in ['right', 'left', 'both', 'neither']:
        raise ValueError("Closed must be one of 'right', 'left', 'both', or 'neither'.")
    if input_stock_prices.empty:
        raise ValueError("Input data must not be empty.")

    # Create a rolling window object
    simple_moving_window = input_stock_prices.rolling(
        window=window_size, 
        min_periods=min_periods, 
        center=window_index, 
        win_type=win_type, 
        on=on, 
        axis=axis, 
        closed=closed)

    # Compute the mean for the rolling window
    simple_moving_average = simple_moving_window.mean()

    return simple_moving_average

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

    if not isinstance(input_stock_prices, pandas.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if window_size is not None and window_size <= 0:
        raise ValueError("Window size must be greater than 0.")
    if min_periods is not None and min_periods < 0:
        raise ValueError("Min_periods must be non-negative.")
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1.")
    if method not in ['single', 'double', 'triple']:
        raise ValueError("Method must be one of 'single', 'double', or 'triple'.")
    if input_stock_prices.empty:
        raise ValueError("Input data must not be empty.")

    # Create an exponential moving window object
    exponential_moving_window = input_stock_prices.ewm(
        com=com,
        span=window_size,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods if min_periods is not None else window_size,
        adjust=decay_adj_factor,
        ignore_na=ignore_na,
        axis=axis,
        times=times,
        method=method)

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
    if closed is not None and closed not in ['right', 'left', 'both', 'neither']:
        raise ValueError("Closed must be one of 'right', 'left', 'both', or 'neither'.")
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

    # Compute the standard deviation for each window
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

    # Create an exponential moving window object
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

def bollinger_bands(input_stock_prices, moving_avg_function: Callable, window_size: int = 50) -> None:
    
    def validate_input(input_stock_prices, window_size):
        # Check if data is either a pandas Series or DataFrame
        if not isinstance(input_stock_prices, (pandas.Series, pandas.DataFrame)):
            raise TypeError("Input data must be either a pandas Series or DataFrame.")

        # If data is a DataFrame, ensure it has only one column
        if isinstance(input_stock_prices, pandas.DataFrame):
            if input_stock_prices.shape[1] != 1:
                raise ValueError("Input DataFrame must contain exactly one column.")
            if input_stock_prices.isnull().values.any():
                raise ValueError("Input DataFrame must not contain any NaN values.")

        # Validate the window size
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer greater than 0.")

        # Check if Series contains any NaN values
        if isinstance(input_stock_prices, pandas.Series) and input_stock_prices.isnull().values.any():
            raise ValueError("Input Series must not contain any NaN values.")

        # Convert Series to DataFrame if necessary
        if isinstance(input_stock_prices, pandas.Series):
            input_stock_prices = pandas.DataFrame({'stock_price': input_stock_prices})

        # Check if index is a datetime index, as it's common in time series data
        if not isinstance(input_stock_prices.index, pandas.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex for time series analysis.")

        return input_stock_prices

    def compute_bollinger_bands(input_stock_prices, moving_avg_function, window_size):
        # Validate the moving average function
        if moving_avg_function not in [simple_moving_average_mean, exponential_moving_average_mean]:
            raise ValueError("Unsupported moving average function.")

        # Compute the moving average
        moving_averages, feature_column = _calculate_moving_average(input_stock_prices, moving_avg_function, window_size)

        # Compute the standard deviation
        std_deviation = _calculate_standard_deviation(input_stock_prices, moving_avg_function, window_size, feature_column)

        # Compute the Bollinger Bands
        bollinger_bands = _calculate_bollinger_bands(moving_averages, std_deviation, window_size)

        return bollinger_bands, feature_column

    def _calculate_moving_average(input_stock_prices, moving_avg_function, window_size):
        bband_moving_averages = moving_average_calculator(input_stock_prices, moving_avg_function, [window_size], plot=False)
        feature_column = input_stock_prices.columns.values[0]
        return bband_moving_averages, feature_column

    def _calculate_standard_deviation(input_stock_prices, moving_avg_function, window_size, feature_column):
        # Determine the standard deviation function based on the moving average type
        standard_deviation_function = simple_moving_average_standard_deviation if moving_avg_function == simple_moving_average_mean else exponential_moving_average_standard_deviation

        # Compute the standard deviation
        bband_standard_deviation = standard_deviation_function(input_stock_prices[feature_column], window_size=window_size)

        return bband_standard_deviation

    def _calculate_bollinger_bands(moving_averages, bband_standard_deviation, window_size):
        window_label = str(window_size) + "-day"

        # Compute the Bollinger Bands
        bollinger_bands = moving_averages.copy()
        bollinger_bands["Lower Band"] = moving_averages[window_label] - (bband_standard_deviation * 2)
        bollinger_bands["Upper Band"] = moving_averages[window_label] + (bband_standard_deviation * 2)

        return bollinger_bands
    
    # Need to do visuals and clean up bbands computation code
    



