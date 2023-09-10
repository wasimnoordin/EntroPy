import numpy
import pandas

from src.performance import (
    calculate_daily_return, 
    calculate_historical_avg_return
)

class Investment:
    """
    Represents a generic financial instrument, such as a stock or financial index.
    Provides methods to compute various financial metrics based on its historical price data.
    """
    
    def __init__(self, asset_price_history: pandas.Series, investment_name: str, investment_category: str = "Financial Index") -> None:
        """
        Initialize the Investment object.
        
        Parameters:
        - asset_price_history: Historical price data of the investment.
        - investment_name: Name of the investment.
        - investment_category: Category of the investment (default is "Financial Index").
        """
        # Assigning the provided data to instance variables
        self.asset_price_history = asset_price_history
        self.investment_name = investment_name
        self.investment_category = investment_category

        # Compute key statistical properties for the investment
        self._compute_investment_properties()

    def _compute_investment_properties(self) -> None:
        """
        Compute and assign key statistical properties like forecast return, 
        volatility, skewness, and kurtosis for the investment.
        """
        # Calculate forecast return for the investment
        self.forecast_investment_return = self.calculate_forecast_investment_return()
        
        # Calculate annualised volatility for the investment
        self.annualised_investment_volatility = self.calculate_annualised_investment_volatility()
        
        # Calculate skewness for the investment's price history
        self.investment_skew = self.calculate_investment_skewness()
        
        # Calculate kurtosis for the investment's price history
        self.investment_kurtosis = self.calculate_investment_kurtosis()

    def calculate_investment_daily_return(self) -> pandas.Series:
        """
        Calculate and return the daily return for the investment based on its historical price data.
        """
        # Check if price history is available
        if self.asset_price_history is None or self.asset_price_history.empty:
            raise ValueError("Price history is not available. Cannot calculate daily returns.")

        try:
            # Calculate daily return using the provided function
            investment_daily_return = calculate_daily_return(self.asset_price_history)
        except Exception as e:
            raise RuntimeError(f"An error occurred while calculating daily returns: {str(e)}")

        return investment_daily_return

    def calculate_forecast_investment_return(self, regular_trading_days: int = 252) -> float:
        """
        Calculate and return the forecasted return for the investment based on its historical price data.
        """
        # Check if price history is available
        if self.asset_price_history is None or self.asset_price_history.empty:
            raise ValueError("Price history is not available. Cannot calculate expected return.")

        try:
            # Calculate forecasted return using the provided function
            forecast_investment_return = calculate_historical_avg_return(self.asset_price_history, regular_trading_days=regular_trading_days)
        except Exception as e:
            raise RuntimeError(f"An error occurred while calculating expected return: {str(e)}")

        return forecast_investment_return

    def calculate_annualised_investment_volatility(self, regular_trading_days: int = 252) -> float:
        """
        Calculate and return the annualised volatility for the investment based on its historical price data.
        """
        # Check if price history is available
        if self.asset_price_history is None or self.asset_price_history.empty:
            raise ValueError("Price history is not available. Cannot calculate volatility.")

        try:
            # Calculate daily returns for the investment
            daily_investment_returns = self.calculate_investment_daily_return()
            
            # Calculate annualised volatility using the standard deviation of daily returns
            annualised_investment_volatility = numpy.sqrt(regular_trading_days) * daily_investment_returns.std()
        except Exception as e:
            raise RuntimeError(f"An error occurred while calculating volatility: {str(e)}")

        return annualised_investment_volatility

    def calculate_investment_skewness(self, skipna=True) -> float:
        """
        Calculate and return the skewness of the investment's historical price data.
        """
        return self.asset_price_history.skew(skipna=skipna)

    def calculate_investment_kurtosis(self, skipna=True) -> float:
        """
        Calculate and return the kurtosis (tailedness) of the investment's historical price data.
        """
        return self.asset_price_history.kurt(skipna=skipna)

    def display_attributes(self):
        """
        Display key attributes of the investment in a tabular format.
        """
        # Define table structure and headers
        separator = "+" + "-" * 40 + "+" + "-" * 60 + "+"
        header = f"| {'Property':<40}| {'Value':<60}|"
        investment_info = f"| {'Investment Name/Category':<40}| {self.investment_category}: {self.investment_name:<60}|"
        forecast_return_info = f"| {'Forecast Return':<40}| {self.forecast_investment_return:<60.4f}|"
        volatility_info = f"| {'Annualised Volatility':<40}| {self.annualised_investment_volatility:<60.4f}|"
        skewness_info = f"| {'Investment Skew':<40}| {self.investment_skew:<60.4f}|"
        kurtosis_info = f"| {'Investment Kurtosis/Tailedness':<40}| {self.investment_kurtosis:<60.4f}|"

        # Construct the table string
        attributes_str = (
            f"{separator}\n"
            f"{header}\n"
            f"{separator}\n"
            f"{investment_info}\n"
            f"{forecast_return_info}\n"
            f"{volatility_info}\n"
            f"{skewness_info}\n"
            f"{kurtosis_info}\n"
            f"{separator}"
        )

        # Print the table
        print(attributes_str)

    def __str__(self) -> str:
        """
        Return a string representation of the Investment object.
        """
        return f"{self.investment_category}: {self.investment_name} with Forecast Return of {self.forecast_investment_return:.4f} and Annualised Volatility of {self.annualised_investment_volatility:.4f}"
