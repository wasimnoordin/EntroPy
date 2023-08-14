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
        self.beta_stocks = pandas.DataFrame(index=["beta coefficient"])
        self.portfolio_distribution = pandas.DataFrame()
        self.stock_objects = {}

        # Return Metrics
        self.average_return = None
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

        # Capital and Allocation
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
        self._add_stock_data(asset_stock)  

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

