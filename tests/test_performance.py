import numpy
import pandas 

from src.performance import (
    calculate_cumulative_return,
    calculate_daily_return,
    calculate_daily_return_proportioned,
    calculate_daily_return_logarithmic,
    calculate_historical_avg_return,
)

def test_calculate_cumulative_return():
    # Generate random data for testing
    l1 = numpy.random.rand(10)
    l2 = numpy.random.rand(10)
    d = {"1": l1, "2": l2}
    df = pandas.DataFrame(d)
    ret = calculate_cumulative_return(df)
    
    # Ensure the function returns a DataFrame and that it's not empty
    assert isinstance(ret, pandas.DataFrame) and not ret.empty
    
    # The first row of the cumulative return should always be 0 (since it's the starting point)
    assert all(ret.iloc[0] == 0.0)
    
    # Validate the cumulative return calculation for the second period
    assert abs(ret.iloc[1]["1"] - (l1[1] - l1[0]) / l1[0]) <= 1e-15

def test_calculate_daily_return():
    # Generate random data for testing
    l1 = numpy.random.rand(5)
    l2 = numpy.random.rand(5)
    d = {"1": l1, "2": l2}
    df = pandas.DataFrame(d)
    ret = calculate_daily_return(df)
    
    # The first row should be NaN since there's no previous day to compare
    assert ret.iloc[0].isnull().all()
    
    # Validate the daily return calculation for subsequent days
    for i in range(1, len(df)):
        assert abs(ret.iloc[i]["1"] - (l1[i] - l1[i-1]) / l1[i-1]) <= 1e-15

def test_calculate_daily_return_proportioned():
    # Generate random data and weights for testing
    l1 = numpy.random.rand(4)
    l2 = numpy.random.rand(4)
    weights = numpy.array([0.25, 0.75])
    d = {"1": l1, "2": l2}
    df = pandas.DataFrame(d)
    ret = calculate_daily_return_proportioned(df, weights)
    
    # Calculate the weighted daily returns manually
    daily_returns = df.pct_change()
    weighted_returns = (daily_returns * weights).sum(axis=1)
    
    # Ensure the calculated returns match the manually computed weighted returns
    assert all(abs(ret[1:] - weighted_returns[1:]) <= 1e-15)

def test_calculate_daily_return_logarithmic():
    # Generate random data for testing
    l1 = numpy.random.rand(5)
    l2 = numpy.random.rand(5)
    d = {"1": l1, "2": l2}
    df = pandas.DataFrame(d)
    ret = calculate_daily_return_logarithmic(df)
    
    # Validate the relationship between logarithmic returns and simple returns
    simple_returns = df.pct_change().dropna()
    assert all(abs(ret.iloc[i]["1"] - numpy.log(1 + simple_returns.iloc[i]["1"])) <= 1e-15 for i in range(len(ret)))

def test_calculate_historical_avg_return():
    # Generate random data for testing
    l1 = numpy.random.rand(100)
    l2 = numpy.random.rand(100)
    d = {"1": l1, "2": l2}
    df = pandas.DataFrame(d)
    ret = calculate_historical_avg_return(df, regular_trading_days=252)

    # Calculate the expected historical average return
    daily_returns = df.pct_change().mean()
    expected = daily_returns * 252
    
    # Validate the calculated historical average return against the expected value
    assert all(abs(ret[col] - expected[col]) <= 1e-12 for col in df.columns)

def test_empty_dataframe_input():
    empty_df = pandas.DataFrame()

    # Test calculate_cumulative_return
    try:
        calculate_cumulative_return(empty_df)
        assert False, "Expected ValueError for empty DataFrame in calculate_cumulative_return"
    except ValueError:
        pass

    # Test calculate_daily_return
    try:
        calculate_daily_return(empty_df)
        assert False, "Expected ValueError for empty DataFrame in calculate_daily_return"
    except ValueError:
        pass

    # Test calculate_daily_return_proportioned
    try:
        calculate_daily_return_proportioned(empty_df, numpy.array([0.5, 0.5]))
        assert False, "Expected ValueError for empty DataFrame in calculate_daily_return_proportioned"
    except ValueError:
        pass

    # Test calculate_daily_return_logarithmic
    try:
        calculate_daily_return_logarithmic(empty_df)
        assert False, "Expected ValueError for empty DataFrame in calculate_daily_return_logarithmic"
    except ValueError:
        pass

    # Test calculate_historical_avg_return
    try:
        calculate_historical_avg_return(empty_df)
        assert False, "Expected ValueError for empty DataFrame in calculate_historical_avg_return"
    except ValueError:
        pass

def test_invalid_types():
    invalid_input = "Not a DataFrame or Series"
    weights = numpy.array([0.5, 0.5])

    # Test calculate_cumulative_return
    try:
        calculate_cumulative_return(invalid_input)
        assert False, "Expected TypeError for invalid input type in calculate_cumulative_return"
    except TypeError:
        pass

    # Test calculate_daily_return
    try:
        calculate_daily_return(invalid_input)
        assert False, "Expected TypeError for invalid input type in calculate_daily_return"
    except (TypeError, AttributeError):
        pass

    # Test calculate_daily_return_proportioned
    try:
        calculate_daily_return_proportioned(invalid_input, weights)
        assert False, "Expected TypeError for invalid input type in calculate_daily_return_proportioned"
    except TypeError:
        pass

 # Test calculate_daily_return_logarithmic
    try:
        calculate_daily_return_logarithmic(invalid_input)
        assert False, "Expected TypeError for invalid input type in calculate_daily_return_logarithmic"
    except (TypeError, AttributeError):
        pass

    # Test calculate_historical_avg_return
    try:
        calculate_historical_avg_return(invalid_input)
        assert False, "Expected ValueError for invalid input type in calculate_historical_avg_return"
    except ValueError:
        pass

def test_infinite_values():
    # Create a DataFrame with infinite values
    infinite_data = pandas.DataFrame({
        "1": [10, 20, numpy.inf, 30],
        "2": [20, numpy.inf, 30, 40]
    })
    weights = numpy.array([0.5, 0.5])

    # Calculate daily return and check if the result contains NaN values due to infinite values in the input
    ret_daily = calculate_daily_return(infinite_data)
    assert ret_daily.isnull().values.any()

    # Calculate proportioned daily return and ensure there are no infinite values in the result
    ret_proportioned = calculate_daily_return_proportioned(infinite_data, weights)
    assert not numpy.isinf(ret_proportioned).any()

    # Replace infinite values with NaN in the input data
    infinite_data.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)

    # Calculate logarithmic daily return and check if the result contains NaN values due to NaN values in the input
    ret_log = calculate_daily_return_logarithmic(infinite_data)
    assert ret_log.isnull().values.any()

    # Calculate historical average return and ensure there are no NaN values in the result
    ret_historical = calculate_historical_avg_return(infinite_data)
    assert not numpy.isnan(ret_historical).any()

def test_constant_price_data():
    # Create a DataFrame with constant prices
    constant_prices = pandas.DataFrame({
        "1": [10, 10, 10, 10],
        "2": [20, 20, 20, 20]
    })
    weights = numpy.array([0.5, 0.5])

    # Calculate cumulative return for constant prices. The return should be 0 after the first value
    ret_cumulative = calculate_cumulative_return(constant_prices)
    assert all(ret_cumulative.iloc[0] == 0.0)
    assert all(ret_cumulative.iloc[1:].sum(axis=1) == 0)

    # Calculate daily return for constant prices. The first row should be NaN and subsequent rows should be 0
    ret_daily = calculate_daily_return(constant_prices)
    assert ret_daily.iloc[0].isnull().all()
    assert all(ret_daily.iloc[1:].sum(axis=1) == 0)

    # Calculate proportioned daily return for constant prices. The return should be 0 for all days
    ret_proportioned = calculate_daily_return_proportioned(constant_prices, weights)
    assert ret_proportioned.iloc[0] == 0.0

    # Calculate logarithmic daily return for constant prices. The return should be 0 for all days
    ret_log = calculate_daily_return_logarithmic(constant_prices)
    assert all(ret_log.iloc[0] == 0.0)
    assert all(ret_log.iloc[1:].sum(axis=1) == 0)

    # Calculate historical average return for constant prices. The return should be 0
    ret_historical = calculate_historical_avg_return(constant_prices)
    assert all(ret_historical == 0)
