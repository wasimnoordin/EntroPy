import numpy
import pandas

def calculate_cumulative_return(input_stock_prices, cumulative_dividend=None):
    if cumulative_dividend is None:
        cumulative_dividend = 0
    
    if not isinstance(input_stock_prices, pandas.DataFrame or pandas.Series):
        raise TypeError("Input data must be a pandas DataFrame.")
    
    if input_stock_prices.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    
    # Filter out rows with any NaN values
    cleaned_data = input_stock_prices.dropna(
        axis=0, 
        how="any", 
        inplace=False
        )
    
    if cleaned_data.empty:
        raise ValueError("All rows in the input DataFrame contain NaN values.")

    initial_prices = cleaned_data.iloc[0]
    dividend_adjusted_prices = cleaned_data + cumulative_dividend
    price_ratios = dividend_adjusted_prices / initial_prices
    cumulative_return = price_ratios - 1.0
    cumulative_return = cumulative_return.astype(
        dtype=numpy.float64, 
        copy=True,
        errors='raise'
    )

    return cumulative_return


def calculate_daily_return(input_stock_prices):
   
    if input_stock_prices.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    
    # Calculate daily returns using pct_change
    daily_return = input_stock_prices.pct_change(
        periods=1, 
        fill_method=None, 
        limit=None, 
        freq=None
        )
    
    # Remove rows with NaN values
    daily_return.dropna(
        axis=0, 
        how="all", 
        inplace=True
        )
    
    # Replace infinite valeus with NaN
    daily_return = daily_return.replace([numpy.inf, -numpy.inf], 
    numpy.nan,  
    limit=None,
    )

    return daily_return


def calculate_daily_return_proportioned(input_stock_prices, allocation):

    if not isinstance(input_stock_prices, pandas.DataFrame):
        raise TypeError("Input stock price must be a pandas DataFrame.")

    if not isinstance(allocation, (numpy.ndarray, pandas.Series)):
        raise TypeError("Weights must be a numpy array or pandas Series.")

    if input_stock_prices.empty or len(allocation) != input_stock_prices.shape[1]:
        raise ValueError("Input DataFrame and weights must have matching dimensions.")

    daily_return = calculate_daily_return(input_stock_prices)
    daily_return_weighted = (daily_return * allocation).sum(
        axis=1, 
        skipna=True, 
        numeric_only=False, 
        min_count=0
        )
    
    return daily_return_weighted

def calculate_daily_return_logarithmic(input_stock_prices):

    # Calculate the daily returns
    percentage_change = calculate_daily_return(input_stock_prices)
    
    # Calculate daily log returns using a lambda function and np.log
    calculate_log = lambda x: numpy.log(1 + x)
    daily_return_logarithmic = percentage_change.apply(
        calculate_log, 
        axis=0, 
        raw=False, 
        result_type=None, 
        args=()
        )
    
    # Remove rows with NaN values
    daily_return_logarithmic = daily_return_logarithmic.dropna(
        axis=0, 
        how="all", 
        inplace=False
        )
    
    return daily_return_logarithmic


def calculate_historical_avg_return(stock_data, regular_trading_days=252):

    if not isinstance(stock_data, (pandas.DataFrame, pandas.Series)):
        raise ValueError("data must be a pandas.DataFrame or pandas.Series")

    daily_returns = calculate_daily_return(stock_data)
    
    if daily_returns.empty:
        raise ValueError("Input DataFrame must contain non-empty data.")

    avg_return = pandas.DataFrame.mean(
        daily_returns,
        axis=0, 
        skipna=True, 
        numeric_only=False
        )
    historical_avg_return = avg_return * regular_trading_days

    return historical_avg_return
