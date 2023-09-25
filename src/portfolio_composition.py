import numpy
import pandas
import datetime
from typing import List

import matplotlib.pyplot as pyplot

from src.portfolio_optimisation import Portfolio_Optimised_Functions
from src.stock import Stock
from src.index import Index

def yfinance_api_invocation(stock_symbols, from_date=None, to_date=None):  
    """
    Invoke the yfinance API to retrieve stock data for given symbols and date range.
    """
    # Check if the from_date is a string and convert it to datetime
    if isinstance(from_date, str):
        from_date = _str_to_datetime(from_date)  
    
    # Check if the to_date is a string and convert it to datetime
    if isinstance(to_date, str):
        to_date = _str_to_datetime(to_date)  
    
    # Use the helper function to fetch stock data
    stock_data = _fetch_yfinance_data(stock_symbols, from_date, to_date)  

    # Adjust the dataframe columns to match the stock symbols
    formatted_stock_data = _adjust_dataframe_cols(stock_data, stock_symbols)  

    return formatted_stock_data

def _str_to_datetime(date_input): 
    """
    Convert a date string to a datetime object.
    """
    # Attempt to convert the string to a datetime object
    try:
        return datetime.datetime.strptime(date_input, "%Y-%m-%d") 
    except ValueError:
        # Raise an error if the date format is not as expected
        raise ValueError(f"Incorrect date format for {date_input}. Expected format: YYYY-MM-DD.")  

def _fetch_yfinance_data(stock_symbols, from_date, to_date): 
    """
    Fetch stock data from yfinance for given symbols and date range.
    """
    try:
        import yfinance
        # Download stock data using yfinance
        return yfinance.download(stock_symbols, start=from_date, end=to_date)  
    except ImportError:
        # Raise an error if yfinance is not installed
        raise ImportError(
            "Please ensure that the package YFinance is installed, as it is prerequisited."
        )
    except Exception as download_error:
        # Raise a general error for any other issues during the data fetch
        raise Exception(
            "An error occurred while fetching stock data from Yahoo Finance via yfinance."
        ) from download_error

def _adjust_dataframe_cols(stock_dataframe, stock_symbols):  
    """
    Adjust the dataframe columns based on the stock names.
    """
    # Check if the dataframe columns need to be adjusted to MultiIndex format
    if len(stock_symbols) > 0 and not isinstance(stock_dataframe.columns, pandas.MultiIndex):
        mindex_col = [(col, stock_symbols[0]) for col in list(stock_dataframe.columns)] 
        stock_dataframe.columns = pandas.MultiIndex.from_tuples(mindex_col, sortorder=None)  
    return stock_dataframe

# ~~~~~~~~~~~ DATA COLUMN MANAGEMENT ~~~~~~~~~~~

def _identify_appropriate_column(stock_quant, stock_symbol, potential_columns, primary_id):
    """
    Identify the appropriate column for a stock symbol within a dataframe.
    """
    # Iterate through potential columns to find the matching column for the stock symbol
    for column in potential_columns:
        # Check if the stock symbol directly matches a column
        if stock_symbol in stock_quant.columns:
            return stock_symbol
        # Check if the columns are in MultiIndex format
        elif isinstance(stock_quant.columns, pandas.MultiIndex):
            # Remove any periods from the column name
            column = column.replace(".", "")
            # Check if the cleaned column name exists in the dataframe
            if column in stock_quant.columns:
                # Update the primary_id list if the column is not already in it
                if column not in primary_id:
                    primary_id.append(column)
                # Check if the stock symbol exists in the sub-columns of the MultiIndex
                if stock_symbol in stock_quant[column].columns:
                    return stock_symbol
                else:
                    # Raise an error if the stock symbol cannot be found in the sub-columns
                    raise ValueError(
                        "Column entries in the second level of the MultiIndex within the pandas DataFrame cannot be located."
                    )
    # Raise an error if no matching column is found for the stock symbol
    raise ValueError("Column identifiers within the provided dataframe could not be located.")

def _format_data_columns(stock_quant, mandatory_column_fields, primary_col_id):
    """
    Format the data columns based on the identified columns.
    """
    # Check if the dataframe columns are in MultiIndex format
    if isinstance(stock_quant.columns, pandas.MultiIndex):
        # Ensure only one primary column ID is used
        if len(primary_col_id) != 1:
            raise ValueError("Presently, the system accommodates only a singular value or quantity per stock.")
        # Return the relevant sub-columns based on the primary column ID
        return stock_quant[primary_col_id[0]].loc[:, mandatory_column_fields]
    else:
        # Return the relevant columns for a regular dataframe
        return stock_quant.loc[:, mandatory_column_fields]

def _fetch_stock_columns(stock_quant, stock_symbols, column_id):
    """
    Fetch the appropriate stock columns based on the stock symbols.
    """
    mandatory_column_fields = []
    primary_col_id = []

    # Iterate through the stock symbols to identify the appropriate columns
    for i in range(len(stock_symbols)):
        column_denomination = _identify_appropriate_column(stock_quant, stock_symbols[i], column_id, primary_col_id)
        mandatory_column_fields.append(column_denomination)

    # Format the dataframe columns based on the identified columns
    stock_quant = _format_data_columns(stock_quant, mandatory_column_fields, primary_col_id)

    # If only one data column per stock exists, rename column labels to the name of the corresponding stock
    renamed_column_mapping = {}
    if len(column_id) == 1:
        for i, symbol in enumerate(stock_symbols):
            renamed_column_mapping.update({(symbol, column_id[0]): symbol})
        stock_quant.rename(columns=renamed_column_mapping, inplace=True)

    return stock_quant

# ~~~~~~~~~~~ PORTFOLIO CONSTRUCTION: YFINANCE API ~~~~~~~~~~~

def _portfolio_assembly_api(
    stock_symbols,
    apportionment=None,
    start=None,
    end=None,
    api_type="yfinance",
    financial_index: str = None, 
):
    """
    Assemble a portfolio using an API.
    """
    pf_api_construct = Portfolio_Optimised_Functions()
    index_data = pandas.DataFrame()
    
    # Fetch stock data from the specified API
    stock_data = _fetch_stock_data_from_api(stock_symbols, start, end, api_type)
    # Fetch index data if a financial index is provided
    index_data = _fetch_index_data(financial_index, start, end, api_type)
    # Determine the final allocation for the stocks
    final_allocation = _get_portfolio_allocation(stock_symbols, apportionment)
    
    pf_api_construct = _portfolio_assembly_df(stock_data, final_allocation, index_data=index_data)
    return pf_api_construct

def _fetch_stock_data_from_api(stock_symbols, start, end, api_type):
    """
    Fetch stock data from the specified API.
    """
    # Check the API type and fetch data accordingly
    if api_type == "yfinance":
        return yfinance_api_invocation(stock_symbols, from_date=start, to_date=end)
    else:
        # Raise an error for unsupported API types
        raise ValueError(f"Unsupported data API: {api_type}")

def _fetch_index_data(index_symbol, start, end, api_type):
    """
    Fetch index data from the specified API.
    """
    # If an index symbol is provided, fetch its data
    if index_symbol:
        return _fetch_stock_data_from_api([index_symbol], start, end, api_type)
    # If no index symbol is provided, return an empty dataframe
    else:
        return pandas.DataFrame()

def _get_portfolio_allocation(stock_symbols, apportionment):
    """
    Determine the allocation for the stocks in the portfolio.
    """
    # If no specific apportionment is provided, generate one
    if apportionment is None:
        return _compose_pf_api_stock_apportionment(column_title=stock_symbols)
    # If apportionment is provided, return it
    return apportionment

# ~~~~~~~~~~~ DATA VERIFICATION & ADJUSTED CLOSE RETRIEVAL ~~~~~~~~~~~

def _verify_stock_presence_in_datacol(stock_symbols, stock_dataframe):
    """
    Verify if the stock symbols are present in the dataframe columns.
    """
    # Check if any of the stock symbols are present in the dataframe columns
    symbols_present = any((symbol in column for symbol in stock_symbols for column in stock_dataframe.columns))
    return symbols_present

def _retrieve_adjusted_close_from_dataframe(stock_df: pandas.DataFrame) -> pandas.Series:
    """
    Retrieve the 'Adjusted Close' data from the provided dataframe.
    """
    # Check if the dataframe has an 'Adj Close' column
    if "Adj Close" not in stock_df.columns:
        raise ValueError("The provided dataframe does not have an 'Adj Close' column.")
    
    # Extract the 'Adj Close' data
    adjust_close_data = stock_df["Adj Close"].squeeze(axis=None)
    return adjust_close_data

# ~~~~~~~~~~~ STOCK APPORTIONMENT: API ~~~~~~~~~~~

def _compose_pf_api_stock_apportionment(column_title=None, stock_df=None):
    """
    Compose the stock apportionment for the portfolio using the API.
    """
    # Ensure only one of 'column_title' or 'stock_df' is provided
    _validate_input_exclusivity(column_title, stock_df)
    # Validate the types of the provided arguments
    _validate_input_types(column_title, stock_df)

    # If a dataframe is provided, extract and validate its column names
    if stock_df:
        column_title = _extract_and_validate_names_from_data(stock_df)
    
    # Generate a balanced allocation for the stocks
    return _generate_balanced_allocation(column_title)

def _generate_balanced_allocation(column_designation):
    """
    Generate a balanced allocation for each stock.
    """
    # Calculate equal allocation for each stock
    allocation = [1.0 / len(column_designation) for _ in column_designation]
    return pandas.DataFrame({"Allocation": allocation, "Name": column_designation})

def _validate_input_exclusivity(column_title, stock_df):
    """
    Ensure only one of 'column_title' or 'stock_df' is provided.
    """
    # Check if both or neither of the arguments are provided
    if (column_title is not None and stock_df is not None) or (column_title is None and stock_df is None):
        raise ValueError("Please ensure to provide either 'column_title' or 'stock_df', but refrain from providing both simultaneously.")

def _validate_input_types(column_title, stock_df):
    """
    Validate the types of provided arguments.
    """
    # Check if 'column_title' is a list
    if column_title and not isinstance(column_title, list):
        raise ValueError("The data type for 'column_title' should be a list.")
    # Check if 'stock_df' is a pandas DataFrame
    if stock_df and not isinstance(stock_df, pandas.DataFrame):
        raise ValueError("The data type for 'stock_df' should be a a pandas.DataFrame.")

def _extract_and_validate_names_from_data(stock_df):
    """
    Extract column names from stock_df and validate them.
    """
    # Extract column names
    column_titles = stock_df.columns
    # Split column names by '-' and strip whitespace
    column_prefixes = [title.split("-")[0].strip() for title in column_titles]
    # Check for conflicting column names
    for x, prefix in enumerate(column_prefixes):
        conflict_prefix = [compar_prefix for position, compar_prefix in enumerate(column_prefixes) if position != x]
        if prefix in conflict_prefix:
            raise ValueError(generate_error_message(prefix))
    return column_titles

# ~~~~~~~~~~~ STOCK APPORTIONMENT: DF ~~~~~~~~~~~

def validate_input_for_allocation(column_title, stock_df):
    """
    Validate the input for the portfolio allocation function.
    """
    # Ensure that either 'names' or 'data' is provided, but not both
    if (column_title is None and stock_df is None) or (column_title is not None and stock_df is not None):
        raise ValueError("Provide either 'names' or 'data', not both.")
    # Validate the type of 'names' argument
    if column_title is not None and not isinstance(column_title, list):
        raise ValueError("The 'names' argument should be a list.")
    # Validate the type of 'data' argument
    if stock_df is not None and not isinstance(stock_df, pandas.DataFrame):
        raise ValueError("The 'data' argument should be a pandas.DataFrame.")

def handle_data_input(stock_df):
    """
    Handle the case where data is provided.
    """
    # Extract column names from the dataframe
    names = stock_df.columns
    # Split column names by '-' and strip whitespace
    splitnames = [name.split("-")[0].strip() for name in names]
    
    # Check for conflicting column names
    for i, splitname in enumerate(splitnames):
        if splitname in splitnames[:i] + splitnames[i+1:]:
            raise ValueError(generate_error_message(splitname))
    return names

def generate_error_message(conflicting_name):
    """
    Generate an error message for conflicting column names.
    """
    # Construct the error message detailing the conflict and potential solutions
    base_message = (
        f"The pandas.DataFrame 'stock_df' displays inconsistency in its column denominations."
        + f" A substring of {conflicting_name} were found in numerous instances, where prefix sharing has occured."
        + "\n Suggested solutions:"
        + "\n 1. Utilize 'formulate_final_portfolio' and offer a 'apportionment' dataframe indicating stock allocations."
        + "\n This approach will aid in isolating accurate columns from the provided data."
        + "\n 2. Ensure the dataframe provided doesn't have columns with similar prefixes, such as 'APPL' and 'APPL - Adj Close'."
    )
    return base_message.format(conflicting_name)

def compute_equal_weights(column_title):
    """
    Compute equal weights for the provided names.
    """
    # Calculate the number of columns
    n = len(column_title)
    # Return a list of equal weights for each column
    return [1.0 / n] * n

def _compose_pf_df_stock_apportionment(column_title=None, stock_df=None):
    """
    Generate a portfolio allocation.
    """
    # Validate the provided input
    validate_input_for_allocation(column_title, stock_df)

    # If a dataframe is provided and it's not empty, handle its input
    if stock_df is not None and not stock_df.empty:
        column_title = handle_data_input(stock_df)

    # Compute equal weights for the columns
    weights = compute_equal_weights(column_title)
    return pandas.DataFrame({"Allocation": weights, "Name": column_title})

# ~~~~~~~~~~~ PORTFOLIO CONSTRUCTION: DF ~~~~~~~~~~~

def _prepare_data(stock_data: pandas.DataFrame, apportionment: pandas.DataFrame, column_id_tags: List[str]) -> pandas.DataFrame:
    """
    Prepare the data by fetching the required stock columns.
    """
    # Verify if the provided stock titles are present in the dataframe
    if not _verify_stock_presence_in_datacol(apportionment['Name'].values, stock_data):
        raise ValueError("Error: None of the provided stock titles were found in the provided dataframe.")
    # Fetch the required stock columns
    return _fetch_stock_columns(stock_data, apportionment['Name'].values, column_id_tags)

def _add_stock_to_portfolio(portfolio: Portfolio_Optimised_Functions, stock_name: str, apportionment_row: pandas.Series, stock_data: pandas.DataFrame) -> None:
    """
    Add an individual stock to the portfolio.
    """
    # Extract the data series for the given stock name
    stock_series = stock_data.loc[:, stock_name].copy(deep=True).squeeze()
    # Create a stock instance with the provided apportionment and data series
    stock_instance = Stock(apportionment_row, asset_price_history=stock_series)
    # Incorporate the stock into the portfolio
    portfolio.incorporate_stock(stock_instance, suspend_changes=True)

def _portfolio_assembly_df(
    stock_data: pandas.DataFrame,
    apportionment: pandas.DataFrame = None,
    column_id_tags: List[str] = None,
    index_data: pandas.DataFrame = None,
) -> Portfolio_Optimised_Functions:
    """
    Assemble the portfolio using the provided dataframe.
    """
    # If no apportionment is provided, generate one
    if apportionment is None:
        apportionment = _compose_pf_df_stock_apportionment(stock_df=stock_data)
    # If no column ID tags are provided, default to "Adj Close"
    if column_id_tags is None:
        column_id_tags = ["Adj Close"]

    # Prepare the stock data
    stock_data = _prepare_data(stock_data, apportionment, column_id_tags)
    
    # Initialize a portfolio object
    portfolio_df = Portfolio_Optimised_Functions()
    
    # If index data is provided and it's not empty, set the financial index for the portfolio
    if index_data is not None and not index_data.empty:
        market_series = _retrieve_adjusted_close_from_dataframe(index_data)
        portfolio_df.financial_index = Index(asset_price_history=market_series)
    
    # Add each stock to the portfolio
    for i in range(len(apportionment)):
        _add_stock_to_portfolio(portfolio_df, apportionment.iloc[i].Name, apportionment.iloc[i], stock_data)
    # Cascade any changes made to the portfolio
    portfolio_df._cascade_changes()
    return portfolio_df

# ~~~~~~~~~~~ SET COMPARISON: UTILITY FUNCTIONS ~~~~~~~~~~~

def _contains_all(set1, set2):
    """
    Check if all elements of set1 are present in set2.
    
    Parameters:
    - set1 (iterable): The first collection of elements.
    - set2 (iterable): The second collection of elements.
    
    Returns:
    - bool: True if all elements of set1 are in set2, otherwise False.
    """
    return set(set1).issubset(set2)

def _contains_at_least_one(set1, set2):
    """
    Check if any element of set1 is present in set2.
    
    Parameters:
    - set1 (iterable): The first collection of elements.
    - set2 (iterable): The second collection of elements.
    
    Returns:
    - bool: True if any element of set1 is in set2, otherwise False.
    """
    return bool(set(set1) & set(set2))

def _complement_of_sets(set1, set2):
    """
    Get elements that are in set2 but not in set1. I.e. the complement.
    
    Parameters:
    - set1 (iterable): The first collection of elements.
    - set2 (iterable): The second collection of elements.
    
    Returns:
    - list: A list containing elements that are in set2 but not in set1.
    """
    return list(set(set2) - set(set1))


# ~~~~~~~~~~~ FINAL PORTFOLIO CONSTRUCTION & ARGS VALIDATION ~~~~~~~~~~~

def validate_kwargs(kwargs, provided_args):
    """
    Validates the input keyword arguments.
    """
    # Reference to the documentation for guidance
    documentation_ref = (
        "Please refer to the report examples for guidance on formatting."
    )
    
    # Check if any arguments were provided
    if not kwargs:
        raise ValueError(f"formulate_final_portfolio() requires input arguments.\n{documentation_ref}")
    
    # Check if any unsupported arguments were provided
    if not _contains_all(kwargs.keys(), provided_args):
        forbidden_arg_ref = _complement_of_sets(provided_args, kwargs.keys())
        raise ValueError(
            f"Unsupported argument detected {forbidden_arg_ref}. Valid arguments include: : {provided_args}. {documentation_ref}"
        )

def check_input_conflict(kwargs, permissable_args, necessary_args, provided_args):
    """
    Check for input argument conflicts.
    """
    # Get the complement of the provided arguments
    complement_input_args = _complement_of_sets(permissable_args, provided_args)
    
    documentation_ref = (
        "Please refer to the report examples for guidance on formatting."
    )
    
    # Check if there are any conflicting arguments
    if _contains_at_least_one(complement_input_args, kwargs.keys()):
        raise ValueError(
            f"Argument conflict detected: {complement_input_args} cannot be used in combination with {necessary_args}. {documentation_ref}"
        )

def handle_portfolio_assembly(kwargs, provided_args):
    """
    Handle the assembly of the portfolio based on the provided arguments.
    """
    fin_portfolio = Portfolio_Optimised_Functions()

    # Handle portfolio assembly via API
    permissible_args_api = ["stock_symbols", "apportionment", "start", "end", "api_type", "financial_index"]
    if _contains_all(["stock_symbols"], kwargs.keys()):
        check_input_conflict(kwargs, permissible_args_api, ["stock_symbols"], provided_args)
        fin_portfolio = _portfolio_assembly_api(**kwargs)

    # Handle portfolio assembly via DataFrame
    permissible_args_df = ["stock_data", "apportionment"]
    if _contains_all(["stock_data"], kwargs.keys()):
        check_input_conflict(kwargs, permissible_args_df, ["stock_data"], provided_args)
        fin_portfolio = _portfolio_assembly_df(**kwargs)

    return fin_portfolio

def validate_final_portfolio(fin_portfolio):
    """
    Validate the attributes of the final portfolio.
    """
    necessary_attributes = [
        (fin_portfolio.portfolio_distribution.empty, 'portfolio_distribution'),
        (fin_portfolio.asset_price_history.empty, 'asset_price_history'),
        (not fin_portfolio.stock_objects, 'stock_objects'),
        (fin_portfolio.pf_forecast_return is None, 'pf_forecast_return'),
        (fin_portfolio.portfolio_volatility is None, 'portfolio_volatility'),
        (fin_portfolio.downside_risk is None, 'downside_risk'),
        (fin_portfolio.sharpe_ratio is None, 'sharpe_ratio'),
        (fin_portfolio.sortino_ratio is None, 'sortino_ratio'),
        (fin_portfolio.portfolio_skewness is None, 'portfolio_skewness'),
        (fin_portfolio.portfolio_kurtosis is None, 'portfolio_kurtosis')
    ]

    # Check for any invalid attributes in the portfolio
    invalid_attrs = [attr_name for is_invalid, attr_name in necessary_attributes if is_invalid]
    if invalid_attrs:
        documentation_ref = (
            "Please refer to the report examples for guidance on formatting."
        )
        raise ValueError(f"Error creating Portfolio instance. Invalid attributes: {invalid_attrs}. {documentation_ref}")

def formulate_final_portfolio(**kwargs):
    """
    Formulate the final portfolio based on the provided arguments.
    """
    provided_args = [
        "apportionment",
        "stock_symbols",
        "start",
        "end",
        "stock_data",
        "api_type",
        "financial_index",
    ]
    
    # Validate the provided keyword arguments
    validate_kwargs(kwargs, provided_args)
    # Assemble the portfolio based on the provided arguments
    fin_portfolio = handle_portfolio_assembly(kwargs, provided_args)
    # Validate the attributes of the final portfolio
    validate_final_portfolio(fin_portfolio)

    return fin_portfolio