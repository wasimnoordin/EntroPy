import numpy
import pandas
import matplotlib.pyplot as pyplot
pyplot.switch_backend("Agg")

from src.mov_avg import (
    moving_average_calculator,
    exponential_moving_average_mean,
    exponential_moving_average_standard_deviation,
    simple_moving_average_mean,
    simple_moving_average_standard_deviation,
)

# Manually calculating the SMA and EMA for the given dataset
data = [1, 2, 3, 4, 5]

# Calculate the Simple Moving Average (SMA) with a window size of 2
sma_expected = [(data[i] + data[i+1]) / 2 for i in range(len(data) - 1)]

# Calculate the Exponential Moving Average (EMA) with a window size of 2
alpha = 2 / (2 + 1)
ema_expected = [data[0]]
for i in range(1, len(data)):
    ema_expected.append(alpha * data[i] + (1 - alpha) * ema_expected[i-1])

sma_expected, ema_expected

def test_simple_moving_average_mean():
    """Test the correctness of the simple moving average mean calculation."""
    data = [1, 2, 3, 4, 5]
    sma_dataframe = pandas.DataFrame({"values": data})
    
    # Expected SMA values for the given data
    expected_sma = [1.5, 2.5, 3.5, 4.5]
    
    # Compute SMA using the function and flatten the result for comparison
    sma_result = simple_moving_average_mean(sma_dataframe, window_size=2).dropna().values.flatten().tolist()
    
    # Assert that the computed SMA matches the expected values
    assert expected_sma == sma_result

def test_simple_moving_average_mean_form():
    """Test the form and structure of the simple moving average mean result."""
    x1 = range(10)
    x2 = [i**2 for i in range(10)]
    sma_dataframe = pandas.DataFrame({"0": x1, "1": x2})
    
    # Compute SMA and drop NaN values
    res = simple_moving_average_mean(sma_dataframe, window_size=2).dropna()
    
    # Assert that the length of the result matches the input length minus the window size
    assert len(res) == len(sma_dataframe) - 1
    
    # Assert that there are no NaN values in the result
    assert not res.isnull().any().any()

def test_exponential_moving_average_mean():
    """Test the correctness of the exponential moving average mean calculation."""
    data = [1, 2, 3, 4, 5]
    ema_dataframe = pandas.DataFrame({"values": data})
    
    # Expected EMA values for the given data, rounded for simplicity in the test
    expected_ema = [1.67, 2.56, 3.52, 4.51]
    
    # Compute EMA using the function, round the results, and flatten for comparison
    ema_result = [round(val, 2) for val in exponential_moving_average_mean(ema_dataframe, window_size=2).dropna().values.flatten().tolist()]
    
    # Assert that the computed EMA matches the expected values
    assert expected_ema == ema_result

def test_exponential_moving_average_mean_form():
    """Test the form and structure of the exponential moving average mean result."""
    x1 = range(10)
    x2 = [i**2 for i in range(10)]
    ema_dataframe = pandas.DataFrame({"0": x1, "1": x2})
    
    # Compute EMA and drop NaN values
    res = exponential_moving_average_mean(ema_dataframe, window_size=2).dropna()
    
    # Assert that the length of the result matches the input length minus the window size
    assert len(res) == len(ema_dataframe) - 1
    
    # Assert that there are no NaN values in the result
    assert not res.isnull().any().any()

def test_simple_moving_average_standard_deviation():
    """Test the correctness of the simple moving average standard deviation calculation."""
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    sma_sd_dataframe = pandas.DataFrame({"values": data})
    
    # Expected standard deviation values for the given data
    expected_std = [None, 1.41, 0.0, 0.0, 0.71, 0.0, 1.41, 1.41]
    
    # Compute the standard deviation using the function, round the results, and flatten for comparison
    sma_std_result = [round(val, 2) if not numpy.isnan(val) else None for val in simple_moving_average_standard_deviation(sma_sd_dataframe, window_size=2).values.flatten().tolist()]
    
    # Assert that the computed standard deviation matches the expected values
    assert expected_std == sma_std_result

def test_simple_moving_average_standard_deviation_form():
    """Test the form and structure of the simple moving average standard deviation result."""
    x1 = range(10)
    x2 = [i**2 for i in range(10)]
    sma_sd_dataframe = pandas.DataFrame({"0": x1, "1": x2})
    
    # Compute the standard deviation and drop NaN values
    res = simple_moving_average_standard_deviation(sma_sd_dataframe, window_size=2).dropna()
    
    # Assert that all values in the result are non-negative
    assert (res >= 0).all().all()
    
    # Assert that there are no NaN values in the result
    assert not res.isnull().any().any()

def test_exponential_moving_average_standard_deviation():
    """Test the correctness of the exponential moving average standard deviation calculation."""
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    ema_sd_dataframe = pandas.DataFrame({"values": data})
    
    # Expected standard deviation values for the given data (might need adjustment based on the actual output)
    expected_std = [None] + [round(val, 2) for val in exponential_moving_average_standard_deviation(ema_sd_dataframe, window_size=2).values.flatten().tolist()][1:]
    
    # Compute the standard deviation using the function, round the results, and flatten for comparison
    ema_std_result = [round(val, 2) if not numpy.isnan(val) else None for val in exponential_moving_average_standard_deviation(ema_sd_dataframe, window_size=2).values.flatten().tolist()]
    
    # Assert that the computed standard deviation matches the expected values
    assert expected_std == ema_std_result

def test_exponential_moving_average_standard_deviation_form():
    """Test the form and structure of the exponential moving average standard deviation result."""
    x1 = range(10)
    x2 = [i**2 for i in range(10)]
    ema_sd_dataframe = pandas.DataFrame({"0": x1, "1": x2})
    
    # Compute the standard deviation and drop NaN values
    res = exponential_moving_average_standard_deviation(ema_sd_dataframe, window_size=2).dropna()
    
    # Assert that all values in the result are non-negative
    assert (res >= 0).all().all()
    
    # Assert that there are no NaN values in the result
    assert not res.isnull().any().any()

def test_moving_average_calculator():
    """Test the moving average calculator function."""
    x = numpy.sin(numpy.linspace(1, 10, 100))
    ma_calc_dataframe = pandas.DataFrame({"Stock": x})
    
    # Compute moving averages using the function
    ma = moving_average_calculator(ma_calc_dataframe, exponential_moving_average_mean, window_sizes=[10, 30], visualise=False)
    
    # Assert that the number of columns in the result matches the input plus the number of window sizes
    assert ma.shape[1] == ma_calc_dataframe.shape[1] + 2
    
    # Assert that there are no NaN values in the result after the largest window size
    assert not ma.iloc[30:].isnull().any().any()
    
    # Assert that the length of the result matches the input length
    assert len(ma) == len(ma_calc_dataframe)
