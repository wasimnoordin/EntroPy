import numpy
import pandas

import matplotlib.pyplot as pyplot
from typing import Callable

from src.mov_avg import(
    simple_moving_average_mean,
    exponential_moving_average_mean,
    simple_moving_average_standard_deviation,
    exponential_moving_average_standard_deviation,
    moving_average_calculator,
)

def bollinger_bands(input_stock_prices, moving_avg_function: Callable, window_size: int = 50) -> None:
    """
    Function to compute the Bollinger Bands for given stock prices.
    
    Parameters:
    - input_stock_prices: The stock prices data, either as a pandas Series or DataFrame.
    - moving_avg_function: Callable function to compute the moving average.
    - window_size: The size of the window for which the moving average is computed. Default is 50.
    """
    
    def validate_input(input_stock_prices, window_size):
        """
        Validates the input stock prices and window size.
        
        Parameters:
        - input_stock_prices: The stock prices data.
        - window_size: The size of the window for moving average.
        
        Returns:
        - Validated input_stock_prices, possibly converted to DataFrame.
        """
        # Check if data is either a pandas Series or DataFrame
        if not isinstance(input_stock_prices, (pandas.Series, pandas.DataFrame)):
            raise TypeError("Input data must be either a pandas Series or DataFrame.")

        # If data is a DataFrame, ensure it has only one column
        if isinstance(input_stock_prices, pandas.DataFrame):
            if input_stock_prices.shape[1] != 1:
                raise ValueError("Input DataFrame must contain exactly one column.")
            # Check for any NaN values in the DataFrame
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
            input_stock_prices = pandas.DataFrame({'Stock Price': input_stock_prices})

        return input_stock_prices

    def compute_bollinger_bands(input_stock_prices, moving_avg_function, window_size):
        """
        Computes the Bollinger Bands for the given stock prices.
        
        Parameters:
        - input_stock_prices: The stock prices data.
        - moving_avg_function: Callable function to compute the moving average.
        - window_size: The size of the window for moving average.
        
        Returns:
        - bollinger_bands: The computed Bollinger Bands.
        - feature_column: The column name used for computation.
        """
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
        """
        Calculate the moving average for the given stock prices.
        
        Parameters:
        - input_stock_prices: DataFrame containing stock prices.
        - moving_avg_function: Callable function to compute the moving average.
        - window_size: The size of the window for which the moving average is computed.
        
        Returns:
        - bband_moving_averages: DataFrame containing the computed moving averages.
        - feature_column: The column name used for computation.
        """
        bband_moving_averages = moving_average_calculator(input_stock_prices, moving_avg_function, [window_size], visualise=False)
        feature_column = input_stock_prices.columns.values[0]
        bband_moving_averages = bband_moving_averages.rename(columns={str(window_size) + 'd': str(window_size) + '-day'})
        return bband_moving_averages, feature_column

    def _calculate_standard_deviation(input_stock_prices, moving_avg_function, window_size, feature_column):
        """
        Calculate the standard deviation for the given stock prices.
        
        Parameters:
        - input_stock_prices: DataFrame containing stock prices.
        - moving_avg_function: Callable function to compute the moving average.
        - window_size: The size of the window for which the moving average is computed.
        - feature_column: The column name used for computation.
        
        Returns:
        - bband_standard_deviation: Series containing the computed standard deviations.
        """
        # Determine the standard deviation function based on the moving average type
        standard_deviation_function = simple_moving_average_standard_deviation if moving_avg_function == simple_moving_average_mean else exponential_moving_average_standard_deviation

        # Compute the standard deviation
        bband_standard_deviation = standard_deviation_function(input_stock_prices[[feature_column]], window_size=window_size)

        return bband_standard_deviation

    def _calculate_bollinger_bands(moving_averages, bband_standard_deviation, window_size):
        """
        Calculate the Bollinger Bands for the given moving averages and standard deviations.
        
        Parameters:
        - moving_averages: DataFrame containing moving averages.
        - bband_standard_deviation: Series containing standard deviations.
        - window_size: The size of the window for which the moving average is computed.
        
        Returns:
        - bollinger_bands: DataFrame containing the computed Bollinger Bands.
        """
        window_label = str(window_size) + "-day"

        # Extract the appropriate column from bband_standard_deviation
        std_deviation_series = bband_standard_deviation.iloc[:, 0]

        # Compute the Bollinger Bands
        upper_bollinger = moving_averages[window_label] + (std_deviation_series * 2)
        lower_bollinger = moving_averages[window_label] - (std_deviation_series * 2)
        
        bollinger_bands = moving_averages.assign(**{
            "Upper BB": upper_bollinger,
            "Lower BB": lower_bollinger
        })

        return bollinger_bands

    def visualise_bollinger_bands(bollinger_bands, input_stock_prices, window_size, moving_avg_function):
        """
        Visualize the Bollinger Bands along with the original stock prices and moving average.
        
        Parameters:
        - bollinger_bands: DataFrame containing the Bollinger Bands.
        - input_stock_prices: DataFrame containing the original stock prices.
        - window_size: The size of the window for which the moving average is computed.
        - moving_avg_function: Callable function to compute the moving average.
        """
        # Create an axis
        _, axes = pyplot.subplots(nrows=1)

        # Define the window label
        window_label = str(window_size) + "-day"

        # Concatenate the upper and lower Bollinger Bands
        x_values = numpy.concatenate([bollinger_bands.index.values, bollinger_bands.index.values[::-1]])
        y_values = numpy.concatenate([bollinger_bands["Upper BB"], bollinger_bands["Lower BB"][::-1]])

        # Fill the area between the upper and lower Bollinger Bands
        axes.fill(x_values, y_values, facecolor="gold", alpha=0.5, label="Bollinger Band")
        
        # Plot the original data and moving average
        input_stock_prices.plot(ax=axes, label='Original Data')
        bollinger_bands[window_label].plot(ax=axes, label=f'{window_size} Day Moving Average')

        # Set the title using the provided function name and window size
        title_text = f"Bollinger Band (+/-2σ) & {moving_avg_function.__name__.replace('_', ' ').title()} over {window_size} Days"
        pyplot.title(title_text)

        # Add legend and customize its appearance
        legend_font = {'size': 10}
        pyplot.legend(prop=legend_font)

        # Set axis labels with custom font size
        xlabel_font = {'size': 12}
        ylabel_font = {'size': 12}
        pyplot.xlabel(input_stock_prices.index.name, fontdict=xlabel_font)
        pyplot.ylabel("Price", fontdict=ylabel_font)

        # Customize the appearance of the plot (optional)
        pyplot.grid(True, linestyle='--', alpha=0.5)  # Adds a grid with dashed lines
        pyplot.tight_layout()  # Adjusts the layout to fit all elements

        # # Display the plot
        # pyplot.show()

    # Validate the input and convert to DataFrame if necessary
    input_stock_prices = validate_input(input_stock_prices, window_size)

    # Compute the Bollinger Bands and feature column
    bollinger_bands, feature_column = compute_bollinger_bands(input_stock_prices, moving_avg_function, window_size)

    # Plot the Bollinger Bands
    visualise_bollinger_bands(bollinger_bands, input_stock_prices, window_size, moving_avg_function)
