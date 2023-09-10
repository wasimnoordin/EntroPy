# Import necessary libraries
import numpy
import pandas
import matplotlib.pyplot as pyplot

# Import specific functions from custom modules
from src.bbands import bollinger_bands
from src.mov_avg import simple_moving_average_mean

# Switch the backend of matplotlib to 'Agg' for rendering plots in non-GUI environments
pyplot.switch_backend("Agg")

# Define a function to test the Bollinger Bands functionality
def test_bollinger_bands():
    # Generate a sine wave data
    x = numpy.sin(numpy.linspace(1, 10, 100))
    
    # Convert the sine wave data into a pandas DataFrame
    df = pandas.DataFrame({"Stock": x}, index=numpy.linspace(1, 10, 100))
    
    # Set the name for the index column
    df.index.name = "Days"
    
    # Calculate and plot the Bollinger Bands for the given data using a simple moving average with a window size of 15
    bollinger_bands(df, simple_moving_average_mean, window_size=15)
    
    # Get the current figure's axis for further inspection
    ax = pyplot.gcf().axes[0]
    
    # Assert to check if there are two lines plotted on the axis (representing the upper and lower Bollinger Bands)
    assert len(ax.lines) == 2  # upper Bollinger Band & lower Bollinger Band
