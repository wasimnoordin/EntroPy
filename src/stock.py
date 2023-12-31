import numpy
import pandas

from src.investment import Investment

class Stock(Investment):

    def __init__(self, stock_details: pandas.DataFrame, asset_price_history: pandas.Series) -> None:
        self.investment_name = stock_details['Name']
        self.stock_details = stock_details
        self.initialize_asset(asset_price_history)
        self.beta_coefficient = None

    def initialize_asset(self, asset_price_history: pandas.Series):
        if not isinstance(asset_price_history, pandas.Series):
            raise TypeError("Data must be a pandas Series containing stock price information.")
        super().__init__(asset_price_history, self.investment_name, investment_category="Stock")

    def calculate_beta_coefficient(self, index_returns: pandas.Series) -> float:

        # Compute daily returns for the stock
        stock_daily_returns = self.calculate_investment_daily_return()

        # Convert market daily returns to DataFrame
        index_returns_dataframe = index_returns.to_frame()[index_returns.name]

        # Compute dispersion matrix between stock and market daily returns
        disp_matrix = numpy.cov(stock_daily_returns, index_returns_dataframe)

        # Extract the dispersion between stock and market, and the variance of the market
        dispersion = disp_matrix[0, 1]
        market_variance = disp_matrix[1, 1]

        # Compute the Beta parameter
        beta_coefficient = dispersion / market_variance

        # Store the Beta value in the object's attribute
        self.beta_coefficient = beta_coefficient

        return beta_coefficient
       
    def display_stock_attributes(self):
        stock_attributes = [
            ("Category", self.investment_category),
            ("Stock", self.investment_name),
            ("Expected Return", f"{self.forecast_investment_return:.4f}"),
            ("Volatility", f"{self.annualised_investment_volatility:.4f}"),
            (f"{self.investment_category} Beta", f"{self.beta_coefficient:.4f}") if self.beta_coefficient else None,
            ("Skewness", f"{self.investment_skew:.4f}"),
            ("Kurtosis", f"{self.investment_kurtosis:.4f}")
        ]

        # Filter out None values (e.g., if beta is None)
        stock_attributes = [prop for prop in stock_attributes if prop]

        # Determine the maximum length of the property names for alignment
        max_length = max(len(prop[0]) for prop in stock_attributes)

        # Construct the upper part with two columns
        upper_part = "=" * (max_length * 2 + 10)
        for i in range(0, len(stock_attributes) - 1, 2):
            upper_part += f"\n{stock_attributes[i][0]}: {stock_attributes[i][1]:<{max_length}} | {stock_attributes[i + 1][0]}: {stock_attributes[i + 1][1]}"
        upper_part += f"\n{stock_attributes[-1][0]}: {stock_attributes[-1][1]}"
        upper_part += "\n" + "=" * (max_length * 2 + 10)

        # Construct the lower part with investment information
        lower_part = "\nINVESTMENT INFORMATION:\n"
        lower_part += "-" * 50
        stock_details_dataframe = self.stock_details.to_frame().transpose()  # Transposing the DataFrame
        stock_name_column = "Stock Name"
        info1_description = stock_details_dataframe.columns[0]  # Getting the column names
        info2_description = stock_details_dataframe.columns[1]
        stock_name = self.investment_name
        info1_value = stock_details_dataframe[info1_description].iloc[0]
        info2_value = stock_details_dataframe[info2_description].iloc[0]
        lower_part += f"\n{stock_name_column:<20} | {info1_description:<15} | {info2_description:>15}"
        lower_part += f"\n{stock_name:<20} | {info1_value:<15} | {info2_value:>15}"
        lower_part += "\n" + "-" * 50

        # Combine upper and lower parts and print the result
        output_string = upper_part + lower_part
        print(output_string)

