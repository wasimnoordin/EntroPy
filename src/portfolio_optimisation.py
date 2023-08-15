import numpy
import pandas
from typing import Union 
import matplotlib.pylab

from src.investment_item import Investment
from src.investment_performance import (
    calculate_cumulative_return,
    calculate_daily_return,
    calculate_daily_return_logarithmic,
    calculate_historical_avg_return,
)
from src.markowitz_efficient_frontier import EfficientFrontierMaster
from src.measures import (
    calculate_stratified_average,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio,
    calculate_annualisation_of_measures,
    calculate_value_at_risk,
    calculate_downside_risk,
    calculate_sortino_ratio,
)
from src.monte_carlo_simulation import MonteCarloMethodology
from src.stock_and_index import (
    Stock, 
    Index,
)

class Portfolio_Optimised_Methods:

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
    def capital_allocation(self, attribute: float) -> None:
        """Sets the capital allocation after validating the input."""
        self.__validate_capital_allocation(attribute)
        self.__capital_allocation = attribute

    def __validate_capital_allocation(self, attribute: float) -> None:
        """Checks the validity of the capital allocation value, ensuring it is a positive number."""
        if attribute is None:
            return

        if not isinstance(attribute, (float, int, numpy.integer, numpy.floating)):
            raise TypeError("The capital allocation must be specified as a number (float or integer).")

        if not attribute > 0:
            raise ValueError("The capital allocation for the portfolio must be a positive value.")

    @regular_trading_days.setter
    def regular_trading_days(self, attribute: int) -> None:
        """Sets the regular trading days after validating the input, and cascades changes to related quantities."""
        self.__validate_regular_trading_days(attribute)
        self.__regular_trading_days = attribute
        # Amend related quantities due to the adjustment in regular trading days
        self._cascade_changes()

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
        self._cascade_changes()

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
    def var_confidence_level(self, attribute: float) -> None:
        self.__validate_confidence_level_value_at_risk(attribute)
        self.__confidence_interval_value_at_risk = attribute
        # now that this changed, update VaR
        self._cascade_changes()

    def __validate_confidence_level_value_at_risk(self, attribute: float) -> None:
        """Validates that the confidence level is a float and within the range [0, 1]."""
        if isinstance(attribute, float):
            # Check if the attribute is within the valid range
            if not (0 < attribute < 1):
                raise ValueError("Confidence level must be a float in the range (0, 1).")
        else:
            raise ValueError("Confidence level must be a float.")
        
    # ~~~~~~~~~~~~~~~~~~~~~~ STOCK MANAGEMENT ~~~~~~~~~~~~~~~~~~~~~~
    
    def add_stock(self, asset_stock: Stock, suspension_changes=False) -> None:
        # Update stocks dictionary
        self._update_stocks(asset_stock)
        
        # Update portfolio with stock information
        self._update_portfolio(asset_stock)
        
        # Update portfolio name
        self._update_portfolio_name()

        # Add stock data to the portfolio
        self._integrate_stock_data(asset_stock)  

        # Only if cascading changes are not suspended, perform them
        if suspension_changes == False:
            self.cascade_changes()

    def _update_stocks(self, asset_stock: Stock) -> None:
        # Update stock_objects dictionary
        self.stock_objects.update({asset_stock.investment_name: asset_stock})

    def _update_portfolio(self, asset_stock: Stock) -> None:
        # Get transposed stock info DataFrame
        stock_info = self._get_stock_info(asset_stock)
        
        # Append stock info to the portfolio
        self.portfolio = self._append_stock_info(stock_info)

    def _update_portfolio_name(self) -> None:
        # Set a descriptive portfolio name
        self.portfolio.investment_name = "Diversified Investment Portfolio"

    def _get_stock_info(self, asset_stock: Stock) -> pandas.DataFrame:
        # Get stock information DataFrame and transpose
        stock_info_frame = asset_stock.stock_details.to_frame()
        stock_info_transposed = stock_info_frame.transpose()
        return stock_info_transposed

    def _append_stock_info(self, stock_info_transposed: pandas.DataFrame) -> pandas.DataFrame:
        # Append stock info to the portfolio DataFrame
        concatenated_portfolio = pandas.concat(
            objs=[self.portfolio, stock_info_transposed],
            axis=0,
            ignore_index=True,
            join='outer'
        )
        return concatenated_portfolio
    
    def _integrate_stock_data(self, asset_stock: Stock) -> None:
        """Integrate stock data into the portfolio object."""
        
        # Append the stock's data to the portfolio's DataFrame
        self._append_stock_to_dataframe(asset_stock)
        
        # Adjust the DataFrame index to match the stock's index
        self._adjust_dataframe_index(asset_stock)
        
        # If a financial index is available, calculate and store the stock's beta parameter
        if self.financial_index:
            self._store_stock_beta(asset_stock)

    def _append_stock_to_dataframe(self, asset_stock: Stock) -> None:
        """Appends the stock data to the portfolio DataFrame."""
        
        # Calculate the location to insert the stock data
        insertion_location = len(self.asset_price_history.columns)
        
        # Get the investment name of the asset_stock
        investment_name = asset_stock.investment_name
        
        # Get the asset_price_history data from the asset_stock
        stock_data = asset_stock.asset_price_history
        
        # Insert the stock data into the portfolio's DataFrame
        self.asset_price_history.insert(
            loc=insertion_location, 
            column=investment_name, 
            value=stock_data,
            allow_duplicates=False
        )

    def _adjust_dataframe_index(self, asset_stock: Stock) -> None:
        """Ensures the DataFrame index is consistent and appropriately named."""
        
        # Set the DataFrame index to match the stock's index values
        self.asset_price_history.set_index(
            asset_stock.asset_price_history.index.values, 
            append=False, 
            inplace=True, 
            verify_integrity=False
        )
        
        # Rename the DataFrame index to "Period"
        self.asset_price_history.index.rename(
            "Period", 
            axis=0, 
            inplace=True, 
            level=None
        )

    def _store_stock_beta(self, asset_stock: Stock) -> None:
        """Calculates and stores the beta parameter for the given stock."""
        
        # Calculate the individual beta using the stock's data and financial index's daily returns
        individual_beta = asset_stock.calculate_pf_beta_coefficient(self.financial_index.calculate_daily_return)
        
        # Create a list containing the individual beta value
        beta_list = [individual_beta]
        
        # Store the list in the beta_dataframe with the investment name as the column
        self.beta_dataframe[asset_stock.investment_name] = beta_list

    def _cascade_changes(self):
        if not self._valid_data_present():
            return
        
        self._update_risk_metrics()
        self._update_return_metrics()
        self._update_beta_metrics()
        self._update_capital_allocation()

    def _valid_data_present(self) -> bool:
        """
        Check if necessary data is present to perform the update.
        
        Returns:
        -- bool: True if valid data is present, False otherwise.
        """
        return not (self.portfolio_distribution.empty or not self.stock_objects or self.asset_price_history.empty)

    def _update_risk_metrics(self):
        """
        Updates the risk metrics of the portfolio.
        """
        self.portfolio_volatility = self.calculate_pf_stock_volatility(regular_trading_days=self.regular_trading_days)
        self.downside_risk = self.calculate_pf_downside_risk(regular_trading_days=self.regular_trading_days)
        self.value_at_risk = self.calculate_pf_value_at_risk()
        self.portfolio_skewness = self.calculate_pf_portfolio_skewness()
        self.portfolio_kurtosis = self.calculate_pf_portfolio_kurtosis()
    
    def _update_return_metrics(self):
        """
        Updates the return metrics of the portfolio.
        """
        self.pf_forecast_return = self.calculate_pf_forecast_return(regular_trading_days=self.regular_trading_days)
        self.sharpe_ratio = self.calculate_pf_sharpe_ratio()
        self.sortino_ratio = self.calculate_pf_sortino_ratio()

    def _update_beta_metrics(self):
        """
        Updates the beta metrics of the portfolio.
        """
        if self.financial_index:
            self.beta_coefficient = self.calculate_pf_beta_coefficient()

    def _update_capital_allocation(self):
        """
        Updates the capital allocation of the portfolio.
        """
        self.capital_allocation = self.portfolio_distribution.Allocation.sum()

    def extract_stock_object (self, investment_name):
        try:
            return self.stock_objects[investment_name]
        except KeyError:
            raise ValueError(f"No stock found with the name: {investment_name}")
        

    def calculate_pf_stock_volatility(self, regular_trading_days: int = 252) -> pandas.Series:
  
        if not isinstance(regular_trading_days, int):
            raise TypeError(f"Expected 'regular_trading_days' to be an integer, but got {type(regular_trading_days).__name__}.")

        if regular_trading_days <= 0:
            raise ValueError("The 'regular_trading_days' should be a positive integer.")
        
        daily_returns = self.calculate_pf_daily_return()
        daily_volatilities = daily_returns.std()
        pf_stock_volatility = daily_volatilities * numpy.sqrt(regular_trading_days)
        
        return pf_stock_volatility
    
    def calculate_pf_daily_return(self):
        return calculate_daily_return(self.asset_price_history)
