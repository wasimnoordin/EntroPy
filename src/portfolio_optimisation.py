import datetime
from typing import Union

import matplotlib.pyplot as pyplot
import numpy 
import pandas

from src.markowitz_efficient_frontier import EfficientFrontierMaster
from src.monte_carlo_simulation import MonteCarloMethodology
from src.measures_common import (
    calculate_stratified_average,
    calculate_portfolio_volatility,
)

from src.measures_ratios import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)

from src.measures_risk import (
    calculate_value_at_risk,
    calculate_downside_risk,
)

from src.performance import (
    calculate_cumulative_return,
    calculate_daily_return_logarithmic,
    calculate_daily_return,
    calculate_historical_avg_return,
)
from src.stock import Stock
from src.index import Index

class Portfolio_Optimised_Functions:

    def __init__(self):
        # DataFrames and Objects
        self.asset_price_history = pandas.DataFrame()
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
        self.beta_dataframe = pandas.DataFrame(index=["beta coefficient"])

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
    def confidence_interval_value_at_risk(self, attribute: float) -> None:
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

    def incorporate_stock(self, stock: Stock, suspend_changes=False) -> None:
        """Add a stock to the portfolio."""
    
        self._store_stock_object(stock)
        self._append_stock_details_to_portfolio(stock)
        self._add_stock_data_to_history(stock)
        
        if not suspend_changes:
            self._cascade_changes()

    def _store_stock_object(self, stock: Stock) -> None:
        """Store the stock object in the dictionary."""
        
        self.stock_objects.update({stock.investment_name: stock})

    def _append_stock_details_to_portfolio(self, stock: Stock) -> None:
        """Append stock details to the portfolio distribution."""
        
        self.portfolio_distribution = pandas.concat(
            [self.portfolio_distribution, stock.stock_details.to_frame().T], ignore_index=True
        )
        self.portfolio_distribution.name = "Allocation of stocks"

    def _add_stock_data_to_history(self, stock: Stock) -> None:
        """Add stock data to the asset price history."""
        
        self._insert_stock_data(stock)
        self._format_asset_history_index(stock)
        self._compute_and_store_beta(stock)

    def _insert_stock_data(self, stock: Stock) -> None:
        """Insert stock data into the asset price history DataFrame."""
        
        self.asset_price_history.insert(
            loc=len(self.asset_price_history.columns), column=stock.investment_name, value=stock.asset_price_history
        )

    def _format_asset_history_index(self, stock: Stock) -> None:
        """Format the index for the asset price history DataFrame."""
        
        self.asset_price_history.set_index(stock.asset_price_history.index.values, inplace=True)
        self.asset_price_history.index.rename("Date", inplace=True)

    def _compute_and_store_beta(self, stock: Stock) -> None:
        """Compute the beta coefficient for the stock and store it."""
        
        if self.financial_index:
            beta_stock = stock.calculate_beta_coefficient(self.financial_index.calculate_daily_return)
            self.beta_dataframe[stock.investment_name] = [beta_stock]

    def _is_portfolio_ready_for_update(self) -> bool:
        """Check if the portfolio is ready for an update."""
        return not (self.portfolio_distribution.empty or not self.stock_objects or self.asset_price_history.empty)

    def _cascade_elementary_metrics(self):
        """Update basic portfolio metrics."""
        self.capital_allocation = self.portfolio_distribution.Allocation.sum()
        self.pf_forecast_return = self.calculate_pf_forecast_return(regular_trading_days=self.regular_trading_days)
        self.portfolio_volatility = self.calculate_pf_volatility(regular_trading_days=self.regular_trading_days)

    def _cascade_risk_metrics(self):
        """Update portfolio risk metrics."""
        self.downside_risk = self.calculate_pf_downside_risk(regular_trading_days=self.regular_trading_days)
        self.value_at_risk = self.calculate_pf_value_at_risk()
    
    def _cascade_performance_metrics(self):
        """Update portfolio performance metrics."""    
        self.sharpe_ratio = self.calculate_pf_sharpe_ratio()
        self.sortino_ratio = self.calculate_pf_sortino_ratio()

    def _cascade_statistical_metrics(self):
        """Update statistical metrics of the portfolio."""
        self.portfolio_skewness = self.calculate_pf_portfolio_skewness()
        self.portfolio_kurtosis = self.calculate_pf_portfolio_kurtosis()

    def _cascade_beta_metrics(self):
        """Update the financial metric of the portfolio."""
        if self.financial_index is not None:
            self.beta_coefficient = self.calculate_pf_beta_coefficient()

    def _cascade_changes(self):
        """Update the portfolio metrics based on category."""
        if self._is_portfolio_ready_for_update():
            self._cascade_elementary_metrics()
            self._cascade_risk_metrics()
            self._cascade_performance_metrics()
            self._cascade_statistical_metrics()
            self._cascade_beta_metrics()

    def extract_stock(self, stock_symbol):
        return self.stock_objects[stock_symbol]
    
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
        """
        Calculate the daily return of the portfolio.
        """
        # Use a helper function to calculate daily return based on asset price history
        return calculate_daily_return(self.asset_price_history)

    def calculate_pf_cumulative_return(self):
        """
        Calculate the cumulative return of the portfolio.
        """
        # Use a helper function to calculate cumulative return based on asset price history
        return calculate_cumulative_return(self.asset_price_history)

    def calculate_pf_daily_return_logarithmic(self):
        """
        Calculate the logarithmic daily return of the portfolio.
        """
        # Use a helper function to calculate logarithmic daily return based on asset price history
        return calculate_daily_return_logarithmic(self.asset_price_history)

    def calculate_pf_historical_avg_return(self, regular_trading_days=252):
        """
        Calculate the historical average return of the portfolio.
        """
        # Use a helper function to calculate historical average return based on asset price history and trading days
        return calculate_historical_avg_return(self.asset_price_history, regular_trading_days=regular_trading_days)

    def calculate_pf_forecast_return(self, regular_trading_days: int = 252) -> float:
        """
        Calculate the forecasted return of the portfolio.
        """
        # Validate the input for regular trading days
        if not isinstance(regular_trading_days, int):
            raise TypeError(f"Expected 'regular_trading_days' to be an integer, but got {type(regular_trading_days).__name__}.")
        if regular_trading_days <= 0:
            raise ValueError("'regular_trading_days' should be a positive integer.")
        
        # Compute portfolio return means based on historical data
        portfolio_historical_return = calculate_historical_avg_return(self.asset_price_history, regular_trading_days=regular_trading_days)

        # Compute portfolio weights
        portfolio_proportions = self.calculate_pf_proportioned_allocation()

        # Calculate the expected return using stratified average
        portfolio_forecast_return = calculate_stratified_average(portfolio_historical_return.values, portfolio_proportions)

        # Assign the calculated forecast return to an instance variable
        self.pf_forecast_return = portfolio_forecast_return
        
        return portfolio_forecast_return

# ~~~~~~~~~~~~~~~~~~~~~~ PORTFOLIO METHODS: BETA & DISPERSION ~~~~~~~~~~~~~~~~~~~~~~

    def calculate_pf_beta_coefficient(self) -> float:
        """
        Calculate the beta coefficient of the portfolio.
        """
        # Determine portfolio allocations
        portfolio_proportions = self.calculate_pf_proportioned_allocation()
        
        # Extract beta values from the dataframe and calculate the weighted mean
        beta_values = self.beta_dataframe.transpose()["beta coefficient"].values
        beta_coefficient = calculate_stratified_average(beta_values, portfolio_proportions)
        
        # Store the calculated Beta as an instance variable
        self.beta_coefficient = beta_coefficient
        
        return beta_coefficient

    def calculate_pf_dispersion_matrix(self) -> pandas.DataFrame:
        """
        Calculate the dispersion matrix of the portfolio.
        """
        # Compute daily returns of the asset price history
        asset_daily_return = calculate_daily_return(self.asset_price_history)
        
        # Compute and return the covariance matrix for the daily returns
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
        """
        Calculate the annualized volatility for each stock in the portfolio.
        """
        # Validate the input: Ensure it's a positive integer
        if not isinstance(regular_trading_days, int):
            raise TypeError(f"Expected 'regular_trading_days' to be an integer, but got {type(regular_trading_days).__name__}.")
        if regular_trading_days <= 0:
            raise ValueError("The 'regular_trading_days' should be a positive integer.")
        
        # Calculate daily returns of the portfolio
        daily_returns = self.calculate_pf_daily_return()
        
        # Calculate daily volatilities for the returns
        daily_volatilities = daily_returns.std()
        
        # Annualize the stock volatilities
        pf_stock_volatility = daily_volatilities * numpy.sqrt(regular_trading_days)
        
        return pf_stock_volatility

    def calculate_pf_volatility(self, regular_trading_days: int = 252) -> float:
        """
        Calculate the annualized volatility of the portfolio.
        """
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
        """
        Calculate the proportioned allocation of the portfolio.
        """
        pf_allocation = self.portfolio_distribution["Allocation"]
        total_allocation = self.capital_allocation
        proportioned_allocation = pf_allocation / total_allocation
        return proportioned_allocation

# ~~~~~~~~~~~~~~~~~~~~~~ PORTFOLIO METHODS: PERFORMANCE ~~~~~~~~~~~~~~~~~~~~~~

    def calculate_pf_sharpe_ratio(self) -> float:
        """
        Calculate the Sharpe Ratio of the portfolio.
        """
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
        """
        Calculate the Sortino Ratio of the portfolio.
        """
        # Calculate Sortino Ratio using the forecast return, downside risk, and risk-free rate of return
        pf_sortino_ratio = calculate_sortino_ratio(
            self.pf_forecast_return, 
            self.downside_risk, 
            self.risk_free_ROR
        )
        return pf_sortino_ratio

# ~~~~~~~~~~~~~~~~~~~~~~ PF OPTIMISATION: MONTE CARLO SIMULATION ~~~~~~~~~~~~~~~~~~~~~~

    def initialise_mcs_instance(self, mcs_iterations=999):
        """
        Initialize or retrieve the Monte Carlo Simulation instance.
        """
        # Check if the Monte Carlo instance exists
        if self.monte_carlo_instance is None:
            # If not, create a new instance with the provided parameters
            self.monte_carlo_instance = MonteCarloMethodology(
                self.calculate_pf_daily_return(),
                mcs_iterations=mcs_iterations,
                risk_free_ROR=self.risk_free_ROR,
                regular_trading_days=self.regular_trading_days,
                seed_allocation=self.calculate_pf_proportioned_allocation().values,
            )
        return self.monte_carlo_instance

    def pf_mcs_optimised_portfolio(self, mcs_iterations=999):
        """
        Optimize the portfolio using Monte Carlo Simulation.
        """
        # Reset the Monte Carlo instance for a fresh optimization
        self.monte_carlo_instance = None

        # Retrieve or initialize the Monte Carlo instance
        if not self.monte_carlo_instance:
            monte_carlo_instance = self.initialise_mcs_instance(mcs_iterations)
            optimal_prop, optimal_prod = monte_carlo_instance.mcs_optimised_portfolio()
            return optimal_prop, optimal_prod

    def pf_mcs_visualisation(self):
        """
        Visualize the results of the Monte Carlo Simulation.
        """
        monte_carlo_instance = self.initialise_mcs_instance()
        monte_carlo_instance.mcs_visualisation()

    def pf_mcs_print_attributes(self):
        """
        Print the attributes of the Monte Carlo Simulation instance.
        """
        monte_carlo_instance = self.initialise_mcs_instance()
        monte_carlo_instance.mcs_print_attributes()

# ~~~~~~~~~~~~~~~~~~~~~~ PF OPTIMISATION: MARKOWITZ EFFICIENT FRONTIER ~~~~~~~~~~~~~~~~~~~~~~

    def initialise_mef_instance(self) -> EfficientFrontierMaster:
        """
        Initialize or retrieve the Markowitz Efficient Frontier instance.
        """
        if not hasattr(self, "efficient_frontier_instance") or self.efficient_frontier_instance is None:
            self.efficient_frontier_instance = EfficientFrontierMaster(
                self.calculate_pf_historical_avg_return(regular_trading_days=1),
                self.calculate_pf_dispersion_matrix(),
                risk_free_ROR=self.risk_free_ROR,
                regular_trading_days=self.regular_trading_days,
            )
        return self.efficient_frontier_instance

    def optimize_pf_mef_volatility_minimisation(self, show_details: bool = False) -> pandas.DataFrame:
        """
        Optimize the portfolio for minimum volatility using the Efficient Frontier.
        """
        # Retrieve or create an instance of EfficientFrontier
        mef_instance = self.initialise_mef_instance()
        
        # Execute the optimization for volatility minimisation
        optimized_proportions = mef_instance.mef_volatility_minimisation()
        
        # If show_details flag is set, display the properties of the efficient frontier optimization
        if show_details:
            mef_instance.mef_metrics(show_details=show_details)
        
        return optimized_proportions

    def optimize_pf_mef_sharpe_maximisation(self, show_details: bool = False) -> pandas.DataFrame:
        """
        Optimize the portfolio for maximum Sharpe Ratio using the Efficient Frontier.
        """
        # Retrieve or create an instance of EfficientFrontier
        mef_instance = self.initialise_mef_instance()

        # Execute the optimization for Sharpe Ratio maximisation
        optimized_mef_proportions = mef_instance.mef_sharpe_maximisation()

        # If show_details flag is set, display the properties of the efficient frontier optimization
        if show_details:
            mef_instance.mef_metrics(show_details=show_details)

        return optimized_mef_proportions

    def optimize_pf_mef_return(self, target_return, show_details=False) -> pandas.DataFrame:
        """
        Optimize the portfolio for a specific target return using the Efficient Frontier.
        """
        # Retrieve or create an instance of EfficientFrontier
        mef_instance = self.initialise_mef_instance()

        # Execute the optimization for the specified target return
        optimal_mef_proportions = mef_instance.mef_return(target_return)

        # If show_details flag is set, display the properties of the efficient frontier optimization
        if show_details:
            mef_instance.mef_metrics(show_details=show_details)

        return optimal_mef_proportions

    def optimize_pf_mef_volatility(self, target_return, show_details=False) -> pandas.DataFrame:
        """
        Optimize the portfolio for a specific target volatility using the Efficient Frontier.
        """
        # Retrieve or create an instance of EfficientFrontier
        mef_instance = self.initialise_mef_instance()

        # Execute the optimization for the specified target volatility
        optimal_mef_proportions = mef_instance.mef_volatility(target_return)

        # If show_details flag is set, display the properties of the efficient frontier optimization
        if show_details:
            mef_instance.mef_metrics(show_details=show_details)

        return optimal_mef_proportions

    def optimize_pf_mef_efficient_frontier(self, target_return=None) -> numpy.ndarray:
        """
        Evaluate the portfolio on the Efficient Frontier.
        """
        # Retrieve or create an instance of EfficientFrontier
        mef_instance = self.initialise_mef_instance()

        # Evaluate the portfolio on the efficient frontier
        frontier_data = mef_instance.mef_evaluate_mef(target_return)

        return frontier_data

    def optimize_pf_plot_mef(self):
        """
        Plot the optimal points on the Efficient Frontier.
        """
        # Retrieve or create an instance of EfficientFrontier
        mef_instance = self.initialise_mef_instance()

        # Plot the optimal points on the efficient frontier
        mef_instance.mef_plot_optimal_mef_points()

    def optimize_pf_plot_vol_and_sharpe_optimal(self):
        """
        Plot the portfolio's volatility and Sharpe Ratio on the Efficient Frontier.
        """
        # Retrieve or create an instance of EfficientFrontier
        mef_instance = self.initialise_mef_instance()

        # Plot the portfolio's volatility and Sharpe Ratio on the efficient frontier
        mef_instance.mef_plot_vol_and_sharpe_optimal()
   
# ~~~~~~~~~~~~~~~~~~~~~~ PF VISUALISATIONS & REPORTING ~~~~~~~~~~~~~~~~~~~~~~

    def pf_stock_visualisation(self, regular_trading_days=252):
        """
        Visualize the annualized returns of stocks against their volatility.
        """
        # Calculate the annual mean returns and volatility of all stocks
        asset_returns = self.calculate_pf_historical_avg_return(regular_trading_days=regular_trading_days)
        asset_volatility = self.calculate_pf_stock_volatility(regular_trading_days=regular_trading_days)

        # Plot the data with enhanced styling
        pyplot.scatter(asset_volatility, asset_returns, marker="8", s=95, color='magenta', edgecolor='black', alpha=0.75)

        # Annotate each stock on the scatter plot
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

        # Set axis labels and a title (only if they haven't been set yet)
        if not pyplot.gca().get_xlabel():
            pyplot.xlabel('Stock Volatility')
        if not pyplot.gca().get_ylabel():
            pyplot.ylabel('Annualised Returns')
        if not pyplot.gca().get_title():
            pyplot.title('Annualised Stock Returns vs. Volatility')

        # Adjust the x and y axis limits for better visualization
        pyplot.xlim([min(pyplot.xlim()[0], asset_volatility.min() - 0.01), max(pyplot.xlim()[1], asset_volatility.max() + 0.01)])
        pyplot.ylim([min(pyplot.ylim()[0], asset_returns.min() - 0.01), max(pyplot.ylim()[1], asset_returns.max() + 0.01)])

    def pf_print_portfolio_attributes(self):
        """
        Print the portfolio's various attributes, including return metrics, risk metrics, distribution metrics, and other metrics.
        """
        stats = self._return_metrics()
        stats += "\n"
        stats += self._risk_metrics()
        stats += "\n"
        stats += self._distribution_metrics() 
        stats += "\n" 
        stats += self._other_metrics()
        stats += "\n"
        print(stats + "\n" + "*" * 100)

    def _return_metrics(self):
        """
        Return the portfolio's return metrics, including forecast return, Sharpe ratio, and Sortino ratio.
        """
        stats = f"📈 Forecast Return: {self.pf_forecast_return:0.3f}\n"
        stats += f"🚀 Sharpe Ratio: {self.sharpe_ratio:0.3f}\n"
        if self.sortino_ratio is not None:
            stats += f"🌪️ Sortino Ratio: {self.sortino_ratio:0.3f}\n"
        else:
            stats += "🌪️ Sortino Ratio: Data not available\n"
        return stats

    def _risk_metrics(self):
        """
        Return the portfolio's risk metrics, including portfolio volatility, downside risk, value at risk, and confidence interval.
        """
        stats = f"🎢 Portfolio Volatility: {self.portfolio_volatility:0.3f}\n"
        stats += f"📉 Downside Risk: {self.downside_risk:0.4f}\n"
        stats += f"❗ Value at Risk: {self.value_at_risk:0.4f}\n"
        stats += f"🔒 Confidence Interval (Value at Risk): {self.confidence_interval_value_at_risk * 100:0.3f} %\n"
        return stats

    def _distribution_metrics(self):
        """
        Return the portfolio's distribution metrics, including skewness, kurtosis, and financial index.
        """
        stats = f"⚖️ Skewness:\n{self.portfolio_skewness}\n"  
        stats += "\n"
        stats += f"🎯 Kurtosis:\n{self.portfolio_kurtosis}\n" 
        stats += "\n" 
        if self.financial_index is not None:
            stats += f"📊 {self.financial_index}\n"
        else:
            stats += "📊 Data not available\n"
        return stats

    def _other_metrics(self):
        """
        Return other metrics of the portfolio, including beta coefficient, trading horizon, and risk-free rate of return.
        """
        stats = ""
        if self.beta_coefficient is not None:
            stats += f"🔄 Beta Coefficient: {self.beta_coefficient:0.3f}\n"
        else:
            stats += "🔄 Beta Coefficient: Data not available\n"
        stats += f"📅 Trading Horizon: {self.regular_trading_days}\n"
        stats += f"💰 Risk Free Rate of Return: {self.risk_free_ROR:.2%}\n"
        return stats

    def __str__(self):
        """
        Return a string representation of the portfolio, listing the stocks it contains.
        """
        stock_id = ', '.join(self.portfolio_distribution.Name.values.tolist())
        return f"Portfolio containing information about stocks: {stock_id}"