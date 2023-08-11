import numpy
import pandas

from src.investment_performance import (
    calculate_daily_return, 
    calculate_historical_avg_return
)

class Investment:
    """
    In the context of this class, "investment" refers to a financial entity
    characterized by its historical price data. It represents a generic 
    financial instrument, which may include a stock or financial index, 
    and the class provides methods to compute various financial metrics such
    as expected return, volatility, skewness, and kurtosis.    
    """
    def __init__(self, price_history: pandas.Series, investment_name: str, investment_category: str = "Financial Index"
    ) -> None:

        self.price_history = price_history
        self.investment_name = investment_name
        self.investment_category = investment_category

        # Compute statistical properties of the investment
        self._compute_investment_properties()

    def _compute_investment_properties(self) -> None:
        """Computes the forecast return, volatility, skewness, and kurtosis of the investment."""
        self.forecast_investment_return = self.calculate_forecast_investment_return()
        self.annualised_investment_volatility = self.calculate_annualised_investment_volatility()
        self.investment_skew = self.calculate_investment_skewness()
        self.investment_kurtosis = self.calculate_investment_kurtosis()

    def calculate_investment_daily_return(self) -> pandas.Series:

        if self.price_history is None or self.price_history.empty:
            raise ValueError("Price history is not available. Cannot calculate daily returns.")

        try:
            investment_daily_return = calculate_daily_return(self.price_history)
        except Exception as e:
            raise RuntimeError(f"An error occurred while calculating daily returns: {str(e)}")

        return investment_daily_return
    
    def calculate_forecast_investment_return(self, regular_trading_days: int = 252) -> float:
        if self.price_history is None or self.price_history.empty:
            raise ValueError("Price history is not available. Cannot calculate expected return.")

        try:
            forecast_investment_return = calculate_historical_avg_return(self.price_history, regular_trading_days=regular_trading_days)
        except Exception as e:
            raise RuntimeError(f"An error occurred while calculating expected return: {str(e)}")

        return forecast_investment_return
    
    def calculate_annualised_investment_volatility(self, regular_trading_days: int = 252) -> float:

        if self.price_history is None or self.price_history.empty:
            raise ValueError("Price history is not available. Cannot calculate volatility.")

        try:
            daily_investment_returns = self.calculate_investment_daily_return()
            annualised_investment_volatility = numpy.sqrt(regular_trading_days, out=None) * daily_investment_returns.std()
        except Exception as e:
            raise RuntimeError(f"An error occurred while calculating volatility: {str(e)}")

        return annualised_investment_volatility

    def calculate_investment_skewness(self, skipna=True) -> float:
        return self.price_history.skew(skipna=skipna)
    
    def calculate_investment_kurtosis(self, skipna=True) -> float:
        return self.price_history.kurt(skipna=skipna)
    
    def display_attributes(self):

        separator = "+" + "-" * 40 + "+" + "-" * 60 + "+"
        header = f"| {'Property':<40}| {'Value':<60}|"
        investment_info = f"| {'Investment Name/Category':<40}| {self.investment_category}: {self.investment_name:<60}|"
        forecast_return_info = f"| {'Forecast Return':<40}| {self.forecast_investment_return:<60.4f}|"
        volatility_info = f"| {'Annualised Volatility':<40}| {self.annualised_investment_volatility:<60.4f}|"
        skewness_info = f"| {'Investment Skew':<40}| {self.investment_skew:<60.4f}|"
        kurtosis_info = f"| {'Investment Kurtosis/Tailedness':<40}| {self.investment_kurtosis:<60.4f}|"

        properties_string = (
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

        print(properties_string)