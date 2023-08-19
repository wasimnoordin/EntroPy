import numpy
import pandas
from typing import Union 
import matplotlib.pylab as pylab
import datetime

from src.investment_item import Investment #NOT NEEDED?
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
            self._cascade_changes()

    def _update_stocks(self, asset_stock: Stock) -> None:
        # Update stock_objects dictionary
        self.stock_objects.update({asset_stock.investment_name: asset_stock})

    def _update_portfolio(self, asset_stock: Stock) -> None:
        # Get transposed stock info DataFrame
        stock_info = self._get_stock_info(asset_stock)
        
        # Append stock info to the portfolio
        self.portfolio_distribution = self._append_stock_info(stock_info)

    def _update_portfolio_name(self) -> None:
        # Set a descriptive portfolio name
        self.portfolio_distribution.investment_name = "Diversified Investment Portfolio"

    def _get_stock_info(self, asset_stock: Stock) -> pandas.DataFrame:
        # Get stock information DataFrame and transpose
        stock_info_frame = asset_stock.stock_details.to_frame()
        stock_info_transposed = stock_info_frame.transpose()
        return stock_info_transposed

    def _append_stock_info(self, stock_info_transposed: pandas.DataFrame) -> pandas.DataFrame:
        # Append stock info to the portfolio DataFrame
        concatenated_portfolio = pandas.concat(
            objs=[self.portfolio_distribution, stock_info_transposed],
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
        self.portfolio_volatility = self.calculate_pf_volatility(regular_trading_days=self.regular_trading_days)
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
        portfolio_dispersion_matrix = asset_daily_return.cov

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
                initial_weights=self.calculate_pf_proportioned_allocation().values,
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
        frontier_data = mef_instance.efficient_frontier(target_return)
        return frontier_data
    
    def optimize_pf_plot_mef(self):
            
            mef_instance = self.initialise_mef_instance()
            mef_instance.plot_optimal_mef_points()

    def optimize_pf_plot_vol_and_sharpe_optimal(self):

        mef_instance = self.initialise_mef_instance()
        mef_instance.plot_vol_and_sharpe_optimal()

# ~~~~~~~~~~~~~~~~~~~~~~ PF VISUALISATIONS & REPORTING ~~~~~~~~~~~~~~~~~~~~~~

    def pf_stock_visualisation(self, regular_trading_days=252):
        # annual mean returns of all stocks
        asset_returns = self.calculate_pf_historical_avg_return(regular_trading_days=regular_trading_days)
        asset_volatility = self.calculate_pf_stock_volatility(regular_trading_days=regular_trading_days)
        
        # Adjusting the size of the plot for better visualization
        pylab.figure(figsize=(10, 6))

        # Plotting the data with enhanced styling
        pylab.scatter(asset_volatility, asset_returns, marker="8", s=120, color='magenta', edgecolor='black', alpha=0.75)

        # Adding gridlines for better clarity
        pylab.grid(True, linestyle='--', alpha=0.75)

        # Annotating the stocks on the scatter plot
        for x, annot_id in enumerate(asset_returns.index, start=0):
            pylab.annotate(
                annot_id,
                (asset_volatility[x], asset_returns[x]),
                xytext=(0, 8),
                textcoords="offset points",
                label=x,
                arrowprops=None,
                annotation_clip=None,
            )

        # Setting axis labels and a title
        pylab.xlabel('Stock Volatility')
        pylab.ylabel('Annualised Returns')
        pylab.title('Annualised Stock Returns vs. Volatility')

        # Adjusting x and y axis limits
        pylab.xlim([asset_volatility.min() - 0.01, asset_volatility.max() + 0.01])
        pylab.ylim([asset_returns.min() - 0.01, asset_returns.max() + 0.01])
    
    def pf_print_portfolio_attributes(self):
        header = self._format_header("Portfolio Attributes")
        stock_info = self._get_stock_info()
        stats = self._get_portfolio_stats()
        skewness = self._get_skewness()
        kurtosis = self._get_kurtosis()
        info = self._get_information()
        
        output = f"\n{header}\n\n{stock_info}\n\n{stats}\n\n{skewness}\n\n{kurtosis}\n\n{info}\n{'=' * 70}\n"
        print(output)

    def _format_header(self, header_text):
        return "📊 " + "=" * 100 + "\n" + header_text.center(100) + "\n" + "=" * 100

    def _get_stock_info(self):
        stock_id = self.portfolio_distribution.Name.values.tolist()
        info = f"📈 Stocks: {', '.join(stock_id)}"
        if self.financial_index is not None:
            info += f"\n🌐 Financial Index: {self.financial_index.investment_name}"
        return info + "\n" + "-" * 100

    def _get_portfolio_stats(self):
        stats = f"📈 Forecast Return: {self.pf_forecast_return:0.3f}"
        stats += f"\n🎢 Portfolio Volatility: {self.portfolio_volatility:0.3f}"
        stats += f"\n🚀 Sharpe Ratio: {self.sharpe_ratio:0.3f}"
        stats += f"\n🌪️ Sortino Ratio: {self.sortino_ratio:0.3f}"
        stats += f"\n📉 Downside Risk: {self.downside_risk:0.4f}"
        stats += f"\n❗ Value at Risk: {self.value_at_risk:0.4f}"
        stats += f"\n🔒 Confidence Interval (Value at Risk): {self.confidence_interval_value_at_risk * 100:0.3f} %"
        if self.beta_coefficient is not None:
            stats += f"\n🔄 Beta Coefficient: {self.beta_coefficient:0.3f}"
        stats += f"\n📅 Trading Horizon: {self.regular_trading_days}"
        stats += f"\n💰 Risk Free Rate of Return: {self.risk_free_ROR:.2%}"
        return stats + "\n" + "*" * 100

    def _get_skewness(self):
        return "🔄 Skewness:\n" + str(self.portfolio_skewness.to_frame().transpose()) + "\n" + "*" * 100

    def _get_kurtosis(self):
        return "📊 Kurtosis:\n" + str(self.portfolio_kurtosis.to_frame().transpose()) + "\n" + "*" * 100

    def _get_information(self):
        self.portfolio_distribution = self.portfolio_distribution.rename(columns={
            'Name': 'Stock Symbol',
            'Weight': 'Allocation',
            'Expected Return': 'Forecasted Return',
            'Volatility': 'Volatility'
             })
        return "ℹ️ Information:\n" + str(self.portfolio_distribution) + "\n" + "=" * 100

    def __str__(self):
        stock_id = ', '.join(self.portfolio_distribution.Name.values.tolist())
        return f"Portfolio containing information about stocks: {stock_id}"

# ~~~~~~~~~~~~~~~~~~~~~~ PF ASSEMBLY & YFINANCE API INTEGRATION ~~~~~~~~~~~~~~~~~~~~~~


    