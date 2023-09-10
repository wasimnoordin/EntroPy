import pandas

from src.performance import (calculate_daily_return, calculate_historical_avg_return)
from src.investment import Investment

class Index(Investment):
  
    def __init__(self, asset_price_history: pandas.Series) -> None:

        # Call the parent class constructor
        super().__init__(asset_price_history, investment_name=asset_price_history.name, investment_category="Financial Index")
        # Compute index returns and store them
        self.calculate_daily_return = self.compute_index_perday_returns()

    def compute_index_perday_returns(self) -> pandas.Series:
        # Utilize the calculate_daily_returns function to compute the daily returns
        index_perday_returns = calculate_daily_return(self.asset_price_history)
        return index_perday_returns
