import numpy
import pandas

from src.investment_performance import (calculate_daily_return, calculate_historical_avg_return)
from src.investment_item import Investment

class Stock(Investment):

    def __init__(self, stock_details: pandas.DataFrame, input_stock_prices: pandas.Series) -> None:
        self.investment_name = stock_details['Name'].iloc[0]
        self.stock_details = stock_details
        self.beta_coefficient = None
        self.initialize_asset(input_stock_prices)

    def initialize_asset(self, input_stock_prices: pandas.Series):
        if not isinstance(input_stock_prices, pandas.Series):
            raise TypeError("Data must be a pandas Series containing stock price information.")
        Investment().__init__(input_stock_prices, self.investment_name, asset_category="Stock")

    def calculate_beta_coefficient(self, index_daily_returns: pandas.Series) -> float:

        # Compute daily returns for the stock
        stock_daily_returns = self.calculate_investment_daily_return()

        # Compute dispersion matrix between stock and market daily returns
        disp_matrix = numpy.cov(stock_daily_returns, index_daily_returns)

        # Extract the dispersion between stock and market, and the variance of the market
        dispersion = disp_matrix[0, 1]
        market_variance = disp_matrix[1, 1]

        # Compute the Beta parameter
        beta_coefficient = dispersion / market_variance

        # Store the Beta value in the object's attribute
        self.beta = beta_coefficient

        return beta_coefficient
    
    def properties(self):
        """Display the stock's properties: Expected Return, Volatility, Beta (optional), Skewness, Kurtosis,
        as well as the Allocation and other information provided in investmentinfo.
        """
        separator = "+" + "-" * 26 + "+" + "-" * 26 + "+" + "-" * 26 + "+"
        header = "|" + f" {self.asset_type.upper()} PROPERTIES: {self.name} ".center(80) + "|"
        properties_string = f"\n{separator}\n{header}\n{separator}\n"

        properties_list = [
            ("Expected Return:", f"{self.expected_return:.3f}"),
            ("Volatility:", f"{self.volatility:.3f}"),
            (f"{self.asset_type} Beta:", f"{self.beta:.3f}"),
            ("Skewness:", f"{self.skew:.5f}"),
            ("Kurtosis:", f"{self.kurtosis:.5f}"),
        ]

        max_property_length = max(len(prop[0]) for prop in properties_list)
        max_value_length = max(len(prop[1]) for prop in properties_list)

        for props in properties_list:
            properties_string += "|" + f"{props[0]:<{max_property_length}}" + "|" + f"{props[1]:>{max_value_length + 1}}" + "|\n"
        
        properties_string += separator
        properties_string += f"\nINVESTMENT INFORMATION:\n"
        investment_info = str(self.investmentinfo.to_frame()).replace('\n', '| ').replace('     ', ' | ')
        properties_string += f"| {investment_info}|\n"
        properties_string += separator
        print(properties_string)




    
