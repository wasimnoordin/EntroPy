import datetime
from typing import List, Union

import matplotlib.pyplot as pyplot
import numpy 
import pandas

from src.markowitz_efficient_frontier import EfficientFrontierMaster
from src.monte_carlo_simulation import MonteCarloMethodology
from src.measures import (
    calculate_downside_risk,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_value_at_risk,
    calculate_stratified_average,
    calculate_portfolio_volatility,
)
from src.investment_performance import (
    calculate_cumulative_return,
    calculate_daily_return_logarithmic,
    calculate_daily_return,
    calculate_historical_avg_return,
)
from src.stock_and_index import Stock, Index

class Portfolio_Optimised_Functions:

    def __init__(self):
        # DataFrames and Objects
        self.asset_price_history = pandas.DataFrame()
        self.beta_dataframe = pandas.DataFrame(index=["beta coefficient"])
        self.portfolio_distribution = pandas.DataFrame()
        self.stock_objects = {}

        # Return Metrics
        self.pf_forecast_return = None
        self.sharpe_ratio = None
        self.sortino_ratio = None

        # Risk Metrics
        self.downside_risk = None
        self.portfolio_volatility = None
        self.portfolio_skewness = None
        self.portfolio_kurtosis = None
        self.value_at_risk = None

        # Beta Metrics
        self.beta_coefficient = None

        # Capital allocation
        self.capital_allocation = None

        # Constants and Settings
        self.confidence_interval_value_at_risk = 0.95
        self.regular_trading_days = 252
        self.risk_free_ROR = 0.005427

        # Instances and Models
        self.efficient_frontier_instance = None
        self.financial_index = None
        self.monte_carlo_instance = None

    # ~~~~~~~~~~~~~~~~~~~~~~ PORTFOLIO GETTERS ~~~~~~~~~~~~~~~~~~~~~~

    @property
    def capital_allocation(self) -> float:
        """Getter for the capital allocation."""
        return self.__capital_allocation
    
    @property
    def regular_trading_days(self) -> int:
        """Getter for annual reg trading days."""
        return self.__regular_trading_days
    
    @property
    def risk_free_ROR(self) -> float:
        """Getter for risk free rate of return."""
        return self.__risk_free_ROR

    @property
    def financial_index(self):
        """Getter for the financial index."""
        return self.__financial_index
    
    @property
    def confidence_interval_value_at_risk(self) -> float:
        """Getter for the confidence interval used in value at risk calculations"""
        return self.__confidence_interval_value_at_risk
      
 # ~~~~~~~~~~~~~~~~~~~~~~ PORTFOLIO SETTERS ~~~~~~~~~~~~~~~~~~~~~~

    @capital_allocation.setter
    def capital_allocation(self, attribute) -> None:
        """Sets the capital allocation after validating the input."""
        if attribute is not None:
            self._validate_capital_allocation(attribute)
            self.__capital_allocation = attribute

    def _validate_capital_allocation(self, attribute: float) -> None:
        """Checks the validity of the capital allocation value, ensuring it is a positive number."""

        if not isinstance(attribute, (float, int, numpy.integer, numpy.floating)):
            raise ValueError("The capital allocation must be specified as a number (float or integer).")

        if not attribute > 0:
            raise ValueError("The capital allocation for the portfolio must be a positive value.")

    @regular_trading_days.setter
    def regular_trading_days(self, attribute: int) -> None:
        """Sets the regular trading days after validating the input, and cascades changes to related quantities."""
        self.__validate_regular_trading_days(attribute)
        self.__regular_trading_days = attribute
        # Amend related quantities due to the adjustment in regular trading days
        self._update()

    def __validate_regular_trading_days(self, attribute: int) -> None:
        """Checks the validity of the regular trading days value, ensuring it is a positive integer."""
        if not isinstance(attribute, int):
            raise ValueError("Regular trading days must be specified as an integer.")
        if not attribute > 0:
            raise ValueError("The number of regular trading days must be greater than zero.")

    @risk_free_ROR.setter
    def risk_free_ROR(self, attribute: Union[float, int]) -> None:
        """Set the risk-free rate with validation and update other quantities."""
        self.__validate_risk_free_ROR(attribute)
        self.__risk_free_ROR = attribute
        # Amend related quantities due to the adjustment in risk_free_ROR
        self._update()

    def __validate_risk_free_ROR(self, attribute: Union[float, int]) -> None:
        """Validates that the risk-free rate is a float or integer and between 0 and 1."""
        # Check for type
        if isinstance(attribute, (float, int)):
            # Check for range
            if 0 <= attribute <= 1:
                # Valid attribute; continue processing
                pass
            else:
                raise ValueError("Risk free rate of return must be between 0 and 1.")
        else:
            raise ValueError("Risk free rate of return must be a float or an integer.")

    @financial_index.setter
    def financial_index(self, index: Index) -> None:
        """Assigns a new financial index to the portfolio.
        This method updates the private attribute __financial_index with the provided Index instance.
        """
        self.__financial_index = index

    @confidence_interval_value_at_risk.setter
    def confidence_interval_value_at_risk(self, attribute: float) -> None:
        self.__validate_confidence_level_value_at_risk(attribute)
        self.__confidence_interval_value_at_risk = attribute
        # now that this changed, update VaR
        self._update()

    def __validate_confidence_level_value_at_risk(self, attribute: float) -> None:
        """Validates that the confidence level is a float and within the range [0, 1]."""
        if isinstance(attribute, float):
            # Check if the attribute is within the valid range
            if not (0 < attribute < 1):
                raise ValueError("Confidence level must be a float in the range (0, 1).")
        else:
            raise ValueError("Confidence level must be a float.")

    def add_stock(self, stock: Stock, defer_update=False) -> None:
        
        # adding stock to dictionary containing all stocks provided
        self.stock_objects.update({stock.investment_name: stock})
        # adding information of stock to the portfolio
        self.portfolio_distribution = pandas.concat(
            [self.portfolio_distribution, stock.stock_details.to_frame().T], ignore_index=True
        )
        # setting an appropriate name for the portfolio
        self.portfolio_distribution.name = "Allocation of stocks"
        # also add stock data of stock to the dataframe
        self._add_stock_data(stock)

        if not defer_update:
            # update quantities of portfolio
            self._update()

    def _add_stock_data(self, stock: Stock) -> None:
        # insert given data into portfolio stocks dataframe:
        self.asset_price_history.insert(
            loc=len(self.asset_price_history.columns), column=stock.investment_name, value=stock.asset_price_history
        )
        # set index correctly
        self.asset_price_history.set_index(stock.asset_price_history.index.values, inplace=True)
        # set index name:
        self.asset_price_history.index.rename("Date", inplace=True)

        if self.financial_index is not None:
            # compute beta parameter of stock
            beta_stock = stock.calculate_beta_coefficient(self.financial_index.calculate_daily_return)
            # add beta of stock to portfolio's betas dataframe
            self.beta_stocks[stock.investment_name] = [beta_stock]

    def _update(self):
        # sanity check (only update values if none of the below is empty):
        if not (self.portfolio_distribution.empty or not self.stock_objects or self.asset_price_history.empty):
            self.capital_allocation = self.portfolio_distribution.Allocation.sum()
            self.expected_return = self.calculate_pf_forecast_return(regular_trading_days=self.regular_trading_days)
            self.volatility = self.calculate_pf_volatility(regular_trading_days=self.regular_trading_days)
            self.downside_risk = self.calculate_pf_downside_risk(regular_trading_days=self.regular_trading_days)
            self.var = self.calculate_pf_value_at_risk()
            self.sharpe = self.calculate_pf_sharpe_ratio()
            self.sortino = self.calculate_pf_sortino_ratio()
            self.skew = self.calculate_pf_portfolio_skewness()
            self.kurtosis = self.calculate_pf_portfolio_kurtosis()
            if self.financial_index is not None:
                self.beta = self.calculate_pf_beta_coefficient()

    def get_stock(self, name):
        return self.stock_objects[name]
    
    # ~~~~~~~~~~~~~~~~~~~~~~ PORTFOLIO METHODS: RISK ~~~~~~~~~~~~~~~~~~~~~~

    def calculate_pf_value_at_risk(self) -> float:

        # Calculate Value at Risk for the portfolio based on its total investment, expected return, volatility, and confidence level
        pf_value_at_risk = calculate_value_at_risk(
            asset_total=self.capital_allocation, 
            asset_average=self.pf_forecast_return, 
            asset_volatility=self.portfolio_volatility, 
            confidence_interval=self.confidence_interval_value_at_risk
        )

        # Store the calculated Value at Risk as an instance variable
        self.value_at_risk = pf_value_at_risk
        
        return pf_value_at_risk
    
    def calculate_pf_downside_risk(self, regular_trading_days: int = 252) -> float:

        # Validate input: Ensure it's a positive integer
        if not isinstance(regular_trading_days, int) or regular_trading_days <= 0:
            raise ValueError("regular_trading_days should be a positive integer.")
        
        # Compute the raw downside risk of the portfolio
        raw_downside_risk = calculate_downside_risk(
            self.asset_price_history, 
            self.calculate_pf_proportioned_allocation(), 
            self.risk_free_ROR
        )
        
        # Annualize the downside risk
        pf_downside_risk = raw_downside_risk * numpy.sqrt(regular_trading_days)
        
        # Store the calculated downside risk as an instance variable
        self.downside_risk = pf_downside_risk
        
        return pf_downside_risk

    # ~~~~~~~~~~~~~~~~~~~~~~ PORTFOLIO METHODS: RETURN ~~~~~~~~~~~~~~~~~~~~~~
    
    def calculate_pf_daily_return(self):
        return calculate_daily_return(self.asset_price_history)
    
    def calculate_pf_cumulative_return(self):
        return calculate_cumulative_return(self.asset_price_history)
    
    def calculate_pf_daily_return_logarithmic(self):
        return calculate_daily_return_logarithmic(self.asset_price_history)
    
    def calculate_pf_historical_avg_return(self, regular_trading_days = 252):
        return calculate_historical_avg_return(self.asset_price_history, regular_trading_days=regular_trading_days)
    
    def calculate_pf_forecast_return(self, regular_trading_days: int = 252) -> float:

        # Validate input
        if not isinstance(regular_trading_days, int):
            raise TypeError(f"Expected 'regular_trading_days' to be an integer, but got {type(regular_trading_days).__name__}.")
        if regular_trading_days <= 0:
            raise ValueError("'regular_trading_days' should be a positive integer.")
        
        # Compute portfolio return means based on historical data
        portfolio_historical_return = calculate_historical_avg_return(self.asset_price_history, regular_trading_days=regular_trading_days)

        # Compute portfolio weights
        portfolio_proportions = self.calculate_pf_proportioned_allocation()

        # Calculate the expected return
        portfolio_forecast_return = calculate_stratified_average(portfolio_historical_return.values, portfolio_proportions)

        # Assign to instance variable
        self.pf_forecast_return = portfolio_forecast_return
        
        return portfolio_forecast_return

    # ~~~~~~~~~~~~~~~~~~~~~~ PORTFOLIO METHODS: BETA & DISPERSION ~~~~~~~~~~~~~~~~~~~~~~
    
    def calculate_pf_beta_coefficient(self) -> float:
        
        # Determine portfolio allocations
        portfolio_proportions = self.calculate_pf_proportioned_allocation()
        
        # Calculate the weighted mean of the Beta values of the stocks in the portfolio
        beta_values = self.beta_dataframe.transpose()["beta"].values
        pf_beta_coefficient = calculate_stratified_average(beta_values, portfolio_proportions)
        
        # Store the calculated Beta as an instance variable
        self.beta_coefficient = pf_beta_coefficient
        
        return pf_beta_coefficient
    
    def calculate_pf_dispersion_matrix(self) -> pandas.DataFrame:
        # Compute daily returns of the asset price history
        asset_daily_return = calculate_daily_return(self.asset_price_history)
        
        # Compute and return the dispersion matrix for the daily returns
        portfolio_dispersion_matrix = asset_daily_return.cov()

        return portfolio_dispersion_matrix

    # ~~~~~~~~~~~~~~~~~~~~~~ PORTFOLIO METHODS: SHAPE ~~~~~~~~~~~~~~~~~~~~~~
    
    def calculate_pf_portfolio_skewness(self) -> pandas.Series:
        """
        Calculates the skewness of the stocks in the portfolio.
        
        Skewness measures the asymmetry of the probability distribution 
        of a real-valued random variable about its mean. A positive skewness 
        indicates a distribution that is skewed towards the right, while a 
        negative skewness indicates a distribution that is skewed to the left.

        Returns:
        - pandas.Series: The skewness values of each stock in the portfolio.
        """
        pf_portfolio_skewness = self.asset_price_history.skew()  # Compute skewness for each stock in the portfolio
        return pf_portfolio_skewness

    def calculate_pf_portfolio_kurtosis(self) -> pandas.Series:
        """
        Calculates the kurtosis of the stocks in the portfolio.
        
        Kurtosis measures the "tailedness" of the probability distribution 
        of a real-valued random variable. High kurtosis indicates a high 
        peak and fat tails, whereas low kurtosis indicates a low peak and 
        thin tails. 

        Returns:
        - pandas.Series: The kurtosis values of each stock in the portfolio.
        """
        pf_portfolio_kurtosis = self.asset_price_history.kurt()  # Compute kurtosis for each stock in the portfolio
        return pf_portfolio_kurtosis
    # ~~~~~~~~~~~~~~~~~~~~~~ PORTFOLIO METHODS: VOLATILITY ~~~~~~~~~~~~~~~~~~~~~~
    
    def calculate_pf_stock_volatility(self, regular_trading_days: int = 252) -> pandas.Series:
  
        if not isinstance(regular_trading_days, int):
            raise TypeError(f"Expected 'regular_trading_days' to be an integer, but got {type(regular_trading_days).__name__}.")

        if regular_trading_days <= 0:
            raise ValueError("The 'regular_trading_days' should be a positive integer.")
        
        daily_returns = self.calculate_pf_daily_return()
        daily_volatilities = daily_returns.std()
        pf_stock_volatility = daily_volatilities * numpy.sqrt(regular_trading_days)
        
        return pf_stock_volatility
    
    def calculate_pf_volatility(self, regular_trading_days: int = 252) -> float:
        
        # Validate the input: Ensure it's a positive integer
        if not isinstance(regular_trading_days, int) or regular_trading_days <= 0:
            raise ValueError("regular_trading_days should be a positive integer.")
        
        # Compute dispersion matrix of the portfolio
        portfolio_dispersion = self.calculate_pf_dispersion_matrix()
        
        # Determine portfolio allocations
        portfolio_proportions = self.calculate_pf_proportioned_allocation()

        # Calculate the raw portfolio's volatility
        raw_portfolio_volatility = calculate_portfolio_volatility(portfolio_dispersion, portfolio_proportions)
        
        # Annualize the portfolio volatility
        annualized_portfolio_volatility = raw_portfolio_volatility * numpy.sqrt(regular_trading_days)
        
        # Store the calculated volatility as an instance variable
        self.portfolio_volatility = annualized_portfolio_volatility
        
        return annualized_portfolio_volatility

    # ~~~~~~~~~~~~~~~~~~~~~~ PORTFOLIO METHODS: CAPITAL ALLOCATION & PROPORTIONMENT ~~~~~~~~~~~~~~~~~~~~~~ 
    
    def calculate_pf_proportioned_allocation(self):
        pf_allocation = self.portfolio_distribution["Allocation"]
        total_allocation = self.capital_allocation
        proportioned_allocation = pf_allocation / total_allocation
        return proportioned_allocation

    # ~~~~~~~~~~~~~~~~~~~~~~ PORTFOLIO METHODS: PERFORMANCE ~~~~~~~~~~~~~~~~~~~~~~
    
    def calculate_pf_sharpe_ratio(self) -> float:

        # Compute the Sharpe Ratio of the portfolio based on its forecast return, volatility, and risk-free rate of return
        pf_sharpe_ratio = calculate_sharpe_ratio(
            self.pf_forecast_return, 
            self.portfolio_volatility, 
            self.risk_free_ROR
        )
        
        # Store the calculated Sharpe Ratio as an instance variable
        self.sharpe_ratio = pf_sharpe_ratio
        
        return pf_sharpe_ratio
    
    def calculate_pf_sortino_ratio(self) -> float:
        # Calculate Sortino Ratio using the forecast return, downside risk, and risk-free rate of return
        pf_sortino_ratio = calculate_sortino_ratio(
            forecast_revenue=self.pf_forecast_return, 
            downside_risk=self.downside_risk, 
            risk_free_ROR=self.risk_free_ROR
        )

        return pf_sortino_ratio

    # ~~~~~~~~~~~~~~~~~~~~~~ PF OPTIMISATION: MONTE CARLO SIMULATION ~~~~~~~~~~~~~~~~~~~~~~
    def initialise_mcs_instance(self, mcs_iterations=999):
        """
        Returns an instance of the MonteCarloMethodology class. If the instance doesn't exist, 
        it initializes and returns a new one.
        """
        if self.monte_carlo_instance is None:
            self.monte_carlo_instance = MonteCarloMethodology(
                self.calculate_pf_daily_return(),
                mcs_iterations=mcs_iterations,
                risk_free_ROR=self.risk_free_ROR,
                regular_trading_days=self.regular_trading_days,
                seed_allocation=self.calculate_pf_proportioned_allocation().values,
            )
        return self.monte_carlo_instance
    
    def pf_mcs_optimised_portfolio(self, mcs_iterations=999):
 
        # Reset Monte Carlo instance for a fresh optimization
        self.monte_carlo_instance = None

        # Retrieve or initialize the Monte Carlo instance
        if not self.monte_carlo_instance:
            monte_carlo_instance = self.initialise_mcs_instance(mcs_iterations)
            optimal_prop, optimal_prod = monte_carlo_instance.mcs_optimised_portfolio()
            return optimal_prop, optimal_prod

    def pf_mcs_visualisation(self):
        """Delegate plotting results to the MonteCarloMethodology instance."""
        monte_carlo_instance = self.initialise_mcs_instance()
        monte_carlo_instance.mcs_visualisation()

    def pf_mcs_print_attributes(self):
        """Retrieve properties from the MonteCarloMethodology instance."""
        monte_carlo_instance = self.initialise_mcs_instance()
        monte_carlo_instance.mcs_print_attributes()
      
    # ~~~~~~~~~~~~~~~~~~~~~~ PF OPTIMISATION: MARKOWITZ EFFICIENT FRONTIER ~~~~~~~~~~~~~~~~~~~~~~

    def initialise_mef_instance(self) -> EfficientFrontierMaster:

        if not hasattr(self, "efficient_frontier_instance") or self.efficient_frontier_instance is None:
            self.efficient_frontier_instance = EfficientFrontierMaster(
                self.calculate_pf_historical_avg_return(regular_trading_days=1),
                self.calculate_pf_dispersion_matrix(),
                risk_free_ROR=self.risk_free_ROR,
                regular_trading_days=self.regular_trading_days,
            )
        return self.efficient_frontier_instance
    
    def optimize_pf_mef_volatility_minimisation(self, show_details: bool = False) -> pandas.DataFrame:

        # Retrieve or create an instance of EfficientFrontier
        mef_instance = self.initialise_mef_instance()
        
        # Execute the optimization for volatility minimisation
        optimized_proportions = mef_instance.mef_volatility_minimisation()
        
        # If show_details flag is set, display the properties of the efficient frontier optimization
        if show_details:
            mef_instance.mef_metrics(show_details=show_details)
        
        return optimized_proportions

    def optimize_pf_mef_sharpe_maximisation(self, show_details: bool = False) -> pandas.DataFrame:

        mef_instance = self.initialise_mef_instance()

        optimized_mef_proportions = mef_instance.mef_sharpe_maximisation()

        if show_details:
            mef_instance.mef_metrics(show_details=show_details)

        return optimized_mef_proportions
    
    def optimize_pf_mef_return(self, target_return, show_details=False) -> pandas.DataFrame:

        mef_instance = self.initialise_mef_instance()
        optimal_mef_proportions = mef_instance.mef_return(target_return)

        if show_details:
            mef_instance.mef_metrics(show_details=show_details)

        return optimal_mef_proportions
    
    def optimize_pf_mef_volatility(self, target_return, show_details=False) -> pandas.DataFrame:

        mef_instance = self.initialise_mef_instance()
        optimal_mef_proportions = mef_instance.mef_volatility(target_return)

        if show_details:
            mef_instance.mef_metrics(show_details=show_details)

        return optimal_mef_proportions

    def optimize_pf_mef_efficient_frontier(self, target_return=None) -> numpy.ndarray:

        mef_instance = self.initialise_mef_instance()
        frontier_data = mef_instance.mef_evaluate_mef(target_return)
        return frontier_data
    
    def optimize_pf_plot_mef(self):
            
            mef_instance = self.initialise_mef_instance()
            mef_instance.mef_plot_optimal_mef_points()

    def optimize_pf_plot_vol_and_sharpe_optimal(self):

        mef_instance = self.initialise_mef_instance()
        mef_instance.mef_plot_vol_and_sharpe_optimal()
    
# ~~~~~~~~~~~~~~~~~~~~~~ PF VISUALISATIONS & REPORTING ~~~~~~~~~~~~~~~~~~~~~~

    def pf_stock_visualisation(self, regular_trading_days=252):
        # annual mean returns of all stocks
        asset_returns = self.calculate_pf_historical_avg_return(regular_trading_days=regular_trading_days)
        asset_volatility = self.calculate_pf_stock_volatility(regular_trading_days=regular_trading_days)

        # Plotting the data with enhanced styling
        pyplot.scatter(asset_volatility, asset_returns, marker="8", s=95, color='magenta', edgecolor='black', alpha=0.75)

        # Annotating the stocks on the scatter plot
        for x, annot_id in enumerate(asset_returns.index, start=0):
            pyplot.annotate(
                annot_id,
                (asset_volatility[x], asset_returns[x]),
                xytext=(0, 8),
                textcoords="offset points",
                label=x,
                arrowprops=None,
                annotation_clip=None,
            )

        # Setting axis labels and a title (only if they haven't been set yet)
        if not pyplot.gca().get_xlabel():
            pyplot.xlabel('Stock Volatility')
        if not pyplot.gca().get_ylabel():
            pyplot.ylabel('Annualised Returns')
        if not pyplot.gca().get_title():
            pyplot.title('Annualised Stock Returns vs. Volatility')

        # Adjusting x and y axis limits
        pyplot.xlim([min(pyplot.xlim()[0], asset_volatility.min() - 0.01), max(pyplot.xlim()[1], asset_volatility.max() + 0.01)])
        pyplot.ylim([min(pyplot.ylim()[0], asset_returns.min() - 0.01), max(pyplot.ylim()[1], asset_returns.max() + 0.01)])


    def pf_print_portfolio_attributes(self):
        stats = self._return_metrics()
        stats += self._risk_metrics()
        stats += self._other_metrics()
        return stats + "\n" + "*" * 100

    def _return_metrics(self):
        stats = f"📈 Forecast Return: {self.pf_forecast_return:0.3f}\n"
        stats += f"🚀 Sharpe Ratio: {self.sharpe_ratio:0.3f}\n"
        if self.sortino_ratio is not None:
            stats += f"🌪️ Sortino Ratio: {self.sortino_ratio:0.3f}\n"
        else:
            stats += "🌪️ Sortino Ratio: Data not available\n"
        return stats

    def _risk_metrics(self):
        stats = f"🎢 Portfolio Volatility: {self.portfolio_volatility:0.3f}\n"
        stats += f"📉 Downside Risk: {self.downside_risk:0.4f}\n"
        stats += f"❗ Value at Risk: {self.value_at_risk:0.4f}\n"
        stats += f"🔒 Confidence Interval (Value at Risk): {self.confidence_interval_value_at_risk * 100:0.3f} %\n"
        return stats

    def _other_metrics(self):
        stats = ""
        if self.beta_coefficient is not None:
            stats += f"🔄 Beta Coefficient: {self.beta_coefficient:0.3f}\n"
        else:
            stats += "🔄 Beta Coefficient: Data not available\n"
        stats += f"📅 Trading Horizon: {self.regular_trading_days}\n"
        stats += f"💰 Risk Free Rate of Return: {self.risk_free_ROR:.2%}\n"
        return stats

    def __str__(self):
        stock_id = ', '.join(self.portfolio_distribution.Name.values.tolist())
        return f"Portfolio containing information about stocks: {stock_id}"


# ~~~~~~~~~~~~~~~~~~~~~~ PF ASSEMBLY & YFINANCE API INTEGRATION ~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~ Main function to fetch and format the stock data ~~~~~~~~~~~

def yfinance_api_invocation(stock_symbols, from_date=None, to_date=None):  

    # Convert string dates to datetime objects if needed
    if isinstance(from_date, str):
        from_date = _str_to_datetime(from_date)  
    
    if isinstance(to_date, str):
        to_date = _str_to_datetime(to_date)  
    
    # Retrieve the stock data
    stock_data = _fetch_yfinance_data(stock_symbols, from_date, to_date)  

    # Adjust the dataframe columns
    formatted_stock_data = _adjust_dataframe_cols(stock_data, stock_symbols)  

    return formatted_stock_data

# Function to convert a date string to a datetime object
def _str_to_datetime(date_input): 
    try:
        return datetime.datetime.strptime(date_input, "%Y-%m-%d") 
    except ValueError:
        raise ValueError(f"Incorrect date format for {date_input}. Expected format: YYYY-MM-DD.")  

# Fetches the stock data from yfinance
def _fetch_yfinance_data(stock_symbols, from_date, to_date): 
    try:
        import yfinance
        return yfinance.download(stock_symbols, start=from_date, end=to_date)  
    except ImportError:
        raise ImportError(
            "Please ensure that the package YFinance is installed, as it is prerequisited."
        )
    except Exception as download_error:
        raise Exception(
            "An error occurred while fetching stock data from Yahoo Finance via yfinance."
        ) from download_error

# ~~~~~~~~~~~~~~~~~~~~~~ TO FIX: PF ASSEMBLY & YFINANCE API INTEGRATION ~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~ Main function to fetch and format the stock data ~~~~~~~~~~~

def yfinance_api_invocation(stock_symbols, from_date=None, to_date=None):  

    # Convert string dates to datetime objects if needed
    if isinstance(from_date, str):
        from_date = _str_to_datetime(from_date)  
    
    if isinstance(to_date, str):
        to_date = _str_to_datetime(to_date)  
    
    # Retrieve the stock data
    stock_data = _fetch_yfinance_data(stock_symbols, from_date, to_date)  

    # Adjust the dataframe columns
    formatted_stock_data = _adjust_dataframe_cols(stock_data, stock_symbols)  

    return formatted_stock_data

# Function to convert a date string to a datetime object
def _str_to_datetime(date_input): 
    try:
        return datetime.datetime.strptime(date_input, "%Y-%m-%d") 
    except ValueError:
        raise ValueError(f"Incorrect date format for {date_input}. Expected format: YYYY-MM-DD.")  

# Fetches the stock data from yfinance
def _fetch_yfinance_data(stock_symbols, from_date, to_date): 
    try:
        import yfinance
        return yfinance.download(stock_symbols, start=from_date, end=to_date)  
    except ImportError:
        raise ImportError(
            "Please ensure that the package YFinance is installed, as it is prerequisited."
        )
    except Exception as download_error:
        raise Exception(
            "An error occurred while fetching stock data from Yahoo Finance via yfinance."
        ) from download_error

# Adjusts the dataframe columns based on the stock names
def _adjust_dataframe_cols(stock_dataframe, stock_symbols):  
    if len(stock_symbols) > 0 and not isinstance(stock_dataframe.columns, pandas.MultiIndex):
        mindex_col = [(col, stock_symbols[0]) for col in list(stock_dataframe.columns)] 
        stock_dataframe.columns = pandas.MultiIndex.from_tuples(mindex_col, sortorder=None)  
    return stock_dataframe

# ~~~~~~~~~~~ Stock Data Column Adjustment Module ~~~~~~~~~~~

def _identify_appropriate_column(stock_quant, stock_symbol, potential_columns, primary_id):
    for column in potential_columns:
        if stock_symbol in stock_quant.columns:
            return stock_symbol
        elif isinstance(stock_quant.columns, pandas.MultiIndex):
            column = column.replace(".", "")
            if column in stock_quant.columns:
                if column not in primary_id:
                    primary_id.append(column)
                if stock_symbol in stock_quant[column].columns:
                    return stock_symbol
                else:
                    raise ValueError(
                        "Column entries in the second level of the MultiIndex within the pandas DataFrame cannot be located."
                    )
    raise ValueError("Column identifiers within the provided dataframe could not be located.")

def _format_data_columns(stock_quant, mandatory_column_fields, primary_col_id):
    if isinstance(stock_quant.columns, pandas.MultiIndex):
        if len(primary_col_id) != 1:
            raise ValueError("Presently, the system accommodates only a singular value or quantity per stock.")
        return stock_quant[primary_col_id[0]].loc[:, mandatory_column_fields]
    else:
        return stock_quant.loc[:, mandatory_column_fields]

def _fetch_stock_columns(stock_quant, stock_symbols, column_id):
    mandatory_column_fields = []
    primary_col_id = []

    for i in range(len(stock_symbols)):
        column_denomination = _identify_appropriate_column(stock_quant, stock_symbols[i], column_id, primary_col_id)
        mandatory_column_fields.append(column_denomination)

    stock_quant = _format_data_columns(stock_quant, mandatory_column_fields, primary_col_id)

    # if only one data column per stock exists, rename column labels
    # to the name of the corresponding stock
    renamed_column_mapping = {}
    if len(column_id) == 1:
        for i, symbol in enumerate(stock_symbols):
            renamed_column_mapping.update({(symbol, column_id[0]): symbol})
        stock_quant.rename(columns=renamed_column_mapping, inplace=True)

    return stock_quant

# ~~~~~~~~~~~ API-Driven Portfolio Construction with YFinance Support and Dynamic Allocation Management ~~~~~~~~~~~

def _portfolio_assembly_api(
    stock_symbols,
    apportionment=None,
    start=None,
    end=None,
    api_type="yfinance",
    index_symbol: str = None,
):
    pf_api_construct = Portfolio_Optimised_Methods()
    index_data = pandas.DataFrame()
    
    stock_data = _fetch_stock_data_from_api(stock_symbols, start, end, api_type)
    index_data = _fetch_index_data(index_symbol, start, end, api_type)
    final_allocation = _get_portfolio_allocation(stock_symbols, apportionment)
    
    pf_api_construct = _portfolio_assembly_df(stock_data, final_allocation, index_data=index_data)
    return pf_api_construct

def _fetch_stock_data_from_api(stock_symbols, start, end, api_type):
    if api_type == "yfinance":
        return yfinance_api_invocation(stock_symbols, from_date=start, to_date=end)
    else:
        raise ValueError(f"Unsupported data API: {api_type}")

def _fetch_index_data(index_symbol, start, end, api_type):
    if index_symbol:
        return _fetch_stock_data_from_api([index_symbol], start, end, api_type)
    else:
        return pandas.DataFrame()

def _get_portfolio_allocation(stock_symbols, apportionment):
    if apportionment is None:
        return _compose_pf_stock_apportionment(column_title=stock_symbols)
    return apportionment

# ~~~~~~~~~~~ Stock Data Verification and Adjusted Close Retrieval ~~~~~~~~~~~

def _verify_stock_presence_in_datacol(stock_symbols, stock_dataframe):
    symbols_present = any((symbol in column for symbol in stock_symbols for column in stock_dataframe.columns))
    return symbols_present

def _retrieve_adjusted_close_from_dataframe(stock_df: pandas.DataFrame) -> pandas.Series:

    if "Adj Close" not in stock_df.columns:
        raise ValueError("The provided dataframe does not have an 'Adj Close' column.")
    
    adjust_close_data = stock_df["Adj Close"].squeeze(axis=None)
    return adjust_close_data

# ~~~~~~~~~~~  Portfolio Stock Apportionment ~~~~~~~~~~~

def _compose_pf_stock_apportionment(column_title=None, stock_df=None):
    _validate_input_exclusivity(column_title, stock_df)
    _validate_input_types(column_title, stock_df)

    if stock_df:
        column_title = _extract_and_validate_names_from_data(stock_df)
    
    return _generate_balanced_allocation(column_title)

def _generate_balanced_allocation(column_designation):
    """Generate balanced allocation for each stock."""
    allocation = [1.0 / len(column_designation) for _ in column_designation]
    return pandas.DataFrame({"Allocation": allocation, "Title": column_designation})

def _validate_input_exclusivity(column_title, stock_df):
    """Ensure only one of 'column_title' or 'stock_df' is provided."""
    if (column_title is not None and stock_df is not None) or (column_title is None and stock_df is None):
        raise ValueError("Please ensure to provide either 'column_title' or 'stock_df', but refrain from providing both simultaneously.")

def _validate_input_types(column_title, stock_df):
    """Validate the types of provided arguments."""
    if column_title and not isinstance(column_title, list):
        raise ValueError("The data type for 'column_title' should be a list.")
    if stock_df and not isinstance(stock_df, pandas.DataFrame):
        raise ValueError("The data type for 'stock_df' should be a a pandas.DataFrame.")

def _extract_and_validate_names_from_data(stock_df):
    """Extract column names from stock_df and validate them."""
    column_titles = stock_df.columns
    column_prefixes = [title.split("-")[0].strip() for title in column_titles]
    for x, prefix in enumerate(column_prefixes):
        conflict_prefix = [compar_prefix for position, compar_prefix in enumerate(column_prefixes) if position != x]
        if prefix in conflict_prefix:
            raise ValueError(
                f"The pandas.DataFrame 'stock_df' displays inconsistency in its column denominations."
                + f" A substring of {prefix} were found in numerous instances, where prefix sharing has occured."
                + "\n Suggested solutions:"
                + "\n 1. Utilize the 'build_portfolio' function and offer a 'apportionment' dataframe indicating stock allocations."
                + "\n This approach will aid in isolating accurate columns from the provided data."
                + "\n 2. Ensure the dataframe provided doesn't have columns with similar prefixes, such as 'APPL' and 'APPL - Adj Close'."
            )
    return column_titles

# ~~~~~~~~~~~ Portfolio Construction and Data Preparation Operations ~~~~~~~~~~~

def _prepare_data(stock_data: pandas.DataFrame, apportionment: pandas.DataFrame, column_id_tags: List[str]) -> pandas.DataFrame:
    """Prepare the data by fetching the required stock columns."""
    if not _verify_stock_presence_in_datacol(apportionment['Name'].values, stock_data):
        raise ValueError("Error: None of the provided stock titles were found in the provided dataframe.")
    return _fetch_stock_columns(stock_data, apportionment['Name'].values, column_id_tags)

def _add_stock_to_portfolio(portfolio: Portfolio_Optimised_Methods, stock_name: str, apportionment_row: pandas.Series, stock_data: pandas.DataFrame) -> None:
    """Add an individual stock to the portfolio."""
    stock_series = stock_data.loc[:, stock_name].copy(deep=True).squeeze()
    stock_instance = Stock(apportionment_row, asset_price_history=stock_series)
    portfolio.incorporate_stock(stock_instance, suspension_changes=True)

def _portfolio_assembly_df(
    stock_data: pandas.DataFrame,
    apportionment: pandas.DataFrame = None,
    column_id_tags: List[str] = None,
    index_data: pandas.DataFrame = None,
) -> Portfolio_Optimised_Methods:
    
    if apportionment is None:
        apportionment = _compose_pf_stock_apportionment(stock_df=stock_data)
    if column_id_tags is None:
        column_id_tags = ["Adj Close"]

    stock_data = _prepare_data(stock_data, apportionment, column_id_tags)
    
    portfolio_df = Portfolio_Optimised_Methods()
    
    if index_data is not None and not index_data.empty:
        market_series = _retrieve_adjusted_close_from_dataframe(index_data)
        portfolio_df.financial_index = Index(price_history=market_series)
    
    for i in range(len(apportionment)):
        _add_stock_to_portfolio(portfolio_df, apportionment.iloc[i].Name, apportionment.iloc[i], stock_data)

    portfolio_df._cascade_changes()
    return portfolio_df

# ~~~~~~~~~~~ Set Comparison Utility Functions ~~~~~~~~~~~

def _all_in(set1, set2):
    """
    Check if all elements of set1 are present in set2.
    
    Parameters:
    - set1 (iterable): The first collection of elements.
    - set2 (iterable): The second collection of elements.
    
    Returns:
    - bool: True if all elements of set1 are in set2, otherwise False.
    """
    return set(set1).issubset(set2)

def _any_in(set1, set2):
    """
    Check if any element of set1 is present in set2.
    
    Parameters:
    - set1 (iterable): The first collection of elements.
    - set2 (iterable): The second collection of elements.
    
    Returns:
    - bool: True if any element of set1 is in set2, otherwise False.
    """
    return bool(set(set1) & set(set2))

def _diff(set1, set2):
    """
    Get elements that are in set2 but not in set1. I.e. the complement.
    
    Parameters:
    - set1 (iterable): The first collection of elements.
    - set2 (iterable): The second collection of elements.
    
    Returns:
    - list: A list containing elements that are in set2 but not in set1.
    """
    return list(set(set2) - set(set1))

# ~~~~~~~~~~~ Final Portfolio Composition, Argument Validation, and Integrity Check ~~~~~~~~~~~

def formulate_final_portfolio(**kwargs):

    # Message encouraging users to consult the function's documentation for guidance.
    documentation_ref = (
        "Please refer to the report examples for guidance on formatting."
    )

    # Message for when an unsupported argument is passed to the function.
    forbidden_arg_err = (
        "Unsupported arguments provided: {}\n"
        "Valid arguments include: {}\n" + documentation_ref
    )

    # Message for when conflicting arguments are provided.
    arg_conflict_err = (
        "Argument conflict detected: {} cannot be used in combination with {}.\n" 
        + documentation_ref
    )
    
    provided_args = [
        "apportionment",
        "stock_symbols",
        "start",
        "end",
        "stock_data",
        "api_type",
        "index_symbol",
    ]
    
    _validate_arguments(kwargs, provided_args, forbidden_arg_err, documentation_ref)

    fin_portfolio = Portfolio_Optimised_Methods()

    if "stock_symbols" in kwargs.keys():
        fin_portfolio = handle_api_assembly(kwargs, provided_args, arg_conflict_err)

    elif "asset_price_history" in kwargs.keys():
        fin_portfolio = handle_df_assembly(kwargs, provided_args, arg_conflict_err)

    _check_portfolio_integrity(fin_portfolio, documentation_ref)

    return fin_portfolio

def _validate_arguments(kwargs, provided_args, forbidden_arg_err, documentation_ref):
    if not kwargs:
        raise ValueError(
            "Error:\nbuild_portfolio() requires input arguments.\n" + documentation_ref
        )
    
    if not _all_in(kwargs.keys(), provided_args):
        forbidden_arg = _diff(provided_args, kwargs.keys())
        raise ValueError(forbidden_arg_err.format(forbidden_arg, provided_args))


def handle_api_assembly(kwargs, provided_args, arg_conflict_err):
    permissible_required_args = ["stock_symbols"]
    permissible_args = [
        "stock_symbols",
        "apportionment",
        "start",
        "end",
        "api_type",
        "index_symbol",
    ]
    complement_args = _diff(permissible_args, provided_args)
    if _all_in(permissible_required_args, kwargs.keys()):
        if _any_in(complement_args, kwargs.keys()):
            raise ValueError(arg_conflict_err.format(complement_args, permissible_required_args))
        
    return _portfolio_assembly_api(**kwargs)

def handle_df_assembly(kwargs, provided_args, arg_conflict_error):
    permissible_required_args = ["asset_price_data"]
    permissible_args = ["asset_price_data", "apportionment"]
    complement_args = _diff(permissible_args, provided_args)
    if _all_in(permissible_required_args, kwargs.keys()):
        if _any_in(complement_args, kwargs.keys()):
            raise ValueError(arg_conflict_error.format(complement_args, permissible_required_args))

    return _portfolio_assembly_df(**kwargs)


def _check_portfolio_integrity(fin_portfolio, documentation_ref):
    if (
        fin_portfolio.portfolio_distribution.empty
        or fin_portfolio.asset_price_history.empty
        or not fin_portfolio.stock_objects
        or any(
            attr is None for attr in [
                fin_portfolio.pf_forecast_return, fin_portfolio.portfolio_volatility, fin_portfolio.downside_risk,
                fin_portfolio.sharpe_ratio, fin_portfolio.sortino_ratio, fin_portfolio.portfolio_skewness, fin_portfolio.portfolio_kurtosis
            ]
        )
    ):
        raise ValueError(
            "This point should not be reached."
            + "An issue occurred during the instantiation of the Portfolio."
            + documentation_ref
        )
