import numpy
import pandas

# Function to calculate the cumulative return of stock prices
def calculate_cumulative_return(input_stock_prices, cumulative_dividend=None):
    # Default cumulative dividend to 0 if not provided
    if cumulative_dividend is None:
        cumulative_dividend = 0
    
    # Validate input type
    if not isinstance(input_stock_prices, pandas.DataFrame or pandas.Series):
        raise TypeError("Input data must be a pandas DataFrame.")
    
    # Check if the input DataFrame is empty
    if input_stock_prices.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    
    # Filter out rows with any NaN values
    cleaned_data = input_stock_prices.dropna(
        axis=0, 
        how="any", 
        inplace=False
        )
    
    # Check if all rows in the cleaned data contain NaN values
    if cleaned_data.empty:
        raise ValueError("All rows in the input DataFrame contain NaN values.")

    # Get the initial prices for comparison
    initial_prices = cleaned_data.iloc[0]
    # Adjust prices with cumulative dividend
    dividend_adjusted_prices = cleaned_data + cumulative_dividend
    # Calculate the price ratios
    price_ratios = dividend_adjusted_prices / initial_prices
    # Calculate the cumulative return
    cumulative_return = price_ratios - 1.0
    # Ensure the return values are of float64 type
    cumulative_return = cumulative_return.astype(
        dtype=numpy.float64, 
        copy=True,
        errors='raise'
    )

    return cumulative_return

# Function to calculate the daily return of stock prices
def calculate_daily_return(input_stock_prices):
   
    # Check if the input DataFrame is empty
    if input_stock_prices.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    
    # Calculate daily returns using pct_change
    daily_return = input_stock_prices.pct_change(
        periods=1, 
        fill_method=None, 
        limit=None, 
        freq=None,
        )
    
    # Remove rows with NaN values
    daily_return.dropna(
        axis=0,
        how='all', 
        inplace=False,
        ignore_index=False,
    )
        
    # Replace infinite values with NaN
    daily_return = daily_return.replace([numpy.inf, -numpy.inf], 
    numpy.nan,  
    limit=None,
    )

    return daily_return

# Function to calculate the proportioned daily return based on given allocation
def calculate_daily_return_proportioned(input_stock_prices, allocation):

    # Validate input types
    if not isinstance(input_stock_prices, pandas.DataFrame):
        raise TypeError("Input stock price must be a pandas DataFrame.")

    if not isinstance(allocation, (numpy.ndarray, pandas.Series)):
        raise TypeError("Weights must be a numpy array or pandas Series.")

    # Check if input DataFrame and weights have matching dimensions
    if input_stock_prices.empty or len(allocation) != input_stock_prices.shape[1]:
        raise ValueError("Input DataFrame and weights must have matching dimensions.")

    # Calculate the daily return
    daily_return = calculate_daily_return(input_stock_prices)
    # Calculate the weighted daily return
    daily_return_weighted = (daily_return * allocation).sum(
        axis=1, 
        skipna=True, 
        numeric_only=False, 
        min_count=0
        )
    
    return daily_return_weighted

# Function to calculate the logarithmic daily return
def calculate_daily_return_logarithmic(input_stock_prices):

    # Calculate the daily returns
    percentage_change = calculate_daily_return(input_stock_prices)
    
    # Calculate daily log returns using a lambda function and numpy.log
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

# Function to calculate the historical average return based on daily returns
def calculate_historical_avg_return(stock_data, regular_trading_days=252):

    # Validate input type
    if not isinstance(stock_data, (pandas.DataFrame, pandas.Series)):
        raise ValueError("data must be a pandas.DataFrame or pandas.Series")

    # Calculate daily returns
    daily_returns = calculate_daily_return(stock_data)
    
    # Check if daily returns DataFrame is non-empty
    if daily_returns.empty:
        raise ValueError("Input DataFrame must contain non-empty data.")

    # Calculate the average return
    avg_return = pandas.DataFrame.mean(
        daily_returns,
        axis=0, 
        skipna=True, 
        numeric_only=False
        )
    # Calculate the historical average return
    historical_avg_return = avg_return * regular_trading_days

    return historical_avg_return
