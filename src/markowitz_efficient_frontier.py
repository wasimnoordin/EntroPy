
import numpy
import pandas
import matplotlib.pylab as pylab

from src import mef_minimisation 
from src.measures_common import calculate_annualisation_of_measures
from scipy import optimize

# Inspired by: 
#         -- https://towardsdatascience.com/efficient-frontier-optimize-portfolio-with-scipy-57456428323e
#         -- https://pub.towardsai.net/portfolio-management-using-python-portfolio-optimization-8a90dd2a21d

class EfficientFrontierInitialization:

    def __init__(
        self, avg_revenue, disp_matrix, risk_free_ROR=0.005427, regular_trading_days=252, method="SLSQP"
    ):
        """
        Initialize the EfficientFrontierInitialization object.
        
        Parameters:
        - avg_revenue: Average revenue for the assets.
        - disp_matrix: Dispersion matrix for the assets.
        - risk_free_ROR: Risk-free rate of return (default is 0.005427).
        - regular_trading_days: Number of trading days in a year (default is 252).
        - method: Optimization method (default is "SLSQP").
        """

        # Validate the provided inputs
        self.validate_inputs(avg_revenue, disp_matrix, risk_free_ROR, regular_trading_days, method)

        # Assign the provided data to instance variables
        self.avg_revenue = avg_revenue
        self.disp_matrix = disp_matrix
        self.risk_free_ROR = risk_free_ROR
        self.regular_trading_days = regular_trading_days
        self.method = method
        self.symbol_stocks = list(avg_revenue.index)
        self.portfolio_size = len(self.symbol_stocks)
        self.prec_opt = ""

        # Set numerical parameters for optimization
        limit = (0, 1)
        self.limits = numpy.full((self.portfolio_size, 2), (0, 1))
        self.initial_guess = numpy.full(self.portfolio_size, 1.0 / self.portfolio_size)
        
        # Define a named function for the constraint
        def constraint_func(x):
            return numpy.sum(x) - 1
        
        # Set the constraint for the optimization
        self.constraints = {"type": "eq", "fun": constraint_func}

        # Placeholder for optimized values/allocations
        self.optimal_mef_points = None
        self.asset_allocation = None
        self.asset_allocation_dataframe = None

    def validate_inputs(self, avg_revenue, disp_matrix, risk_free_ROR, regular_trading_days, method):
        """
        Validate the provided inputs for the initialization.
        """
        # List of supported optimization solvers
        supported_solvers = [
            "BFGS", "CG", "COBYLA", "dogleg", "L-BFGS-B", 
            "Nelder-Mead", "Newton-CG", "Powell", "SLSQP", "TNC",
            "trust-constr", "trust-exact", "trust-krylov", "trust-ncg"
        ]
        
        # Validate each input with appropriate checks
        if not isinstance(avg_revenue, pandas.Series) or avg_revenue.empty:
            raise ValueError("avg_revenue is required as a non-empty pandas.Series.")
        if not isinstance(disp_matrix, pandas.DataFrame) or disp_matrix.empty:
            raise ValueError("disp_matrix is requried as a non-empty pandas.DataFrame")
        if len(avg_revenue) != len(disp_matrix):
            raise ValueError("avg_revenue and disp_matrix must have matching dimensions.")
        if not isinstance(risk_free_ROR, (int, float)) or risk_free_ROR < 0:
            raise ValueError("risk_free_ROR is expected to be a non-negative integer or float.")
        if not isinstance(regular_trading_days, int) or regular_trading_days <= 0:
            raise ValueError("regular_trading_days is expected to be a positive integer.")
        if not isinstance(method, str):
            raise ValueError("The input for the solver is required to be in string format.")
        if method not in supported_solvers:
            raise ValueError("The provided method is not compatible with minimize function from scipy.optimize")
        
    def _asset_allocation_dframe(self, allocation):
        """
        Create a DataFrame to represent asset allocation.
        
        Parameters:
        - allocation: Allocation values for the assets.
        
        Returns:
        - A pandas DataFrame representing the asset allocation.
        """
        # Check if allocation is a numpy.ndarray
        if not isinstance(allocation, numpy.ndarray):
            raise ValueError("allocation is expected to be a numpy.ndarray")

        # Create a pandas DataFrame with the allocation, indexed by asset designations
        # and with a single column named "Allocation"
        return pandas.DataFrame(allocation, index=self.symbol_stocks, columns=["Allocation"])


class EfficientFrontierOptimization:

    def __init__(self, initialization):
        self.initialization = initialization
    
    def _asset_allocation_dframe(self, allocation):
        """
        Create a DataFrame to represent asset allocation using the initialization's method.
        
        Parameters:
        - allocation: Allocation values for the assets.
        
        Returns:
        - A pandas DataFrame representing the asset allocation.
        """

        return self.initialization._asset_allocation_dframe(allocation)

    def mef_volatility_minimisation(self, record_optimized_allocation=True):
        """
        Optimize the portfolio to minimize volatility.
        
        Parameters:
        - record_optimized_allocation: Boolean indicating whether to record the optimized allocation.
        
        Returns:
        - A DataFrame of asset allocations if record_optimized_allocation is True, otherwise the optimal allocations.
        """

        # Validate that recording the optmized asset allocations is a boolean, as it controls whether the allocation are updated or not
        if not isinstance(record_optimized_allocation, bool):
            raise TypeError("The variable record_optimized_allocation should be a boolean value.")

        # Extract average revenue and dispersion matrix values for optimization
        avg_revenue = self.initialization.avg_revenue.values
        disp_matrix = self.initialization.disp_matrix.values

        # Extract other parameters required for optimization
        initial_guess = self.initialization.initial_guess
        method = self.initialization.method
        limits = self.initialization.limits
        constraints = self.initialization.constraints

        # Define the parameters for the optimization function
        param_opt = (avg_revenue, disp_matrix)

        # Perform the optimization using SciPy's minimize function
        # The objective function is calculate_annualized_volatility
        # The "x" in the result will contain the values that minimize the volatility
        output = optimize.minimize(
            mef_minimisation.calculate_annualized_volatility,
            args=param_opt,
            x0=initial_guess,
            method=method,
            bounds=limits,
            constraints=constraints,
        )

        # Update the preceding optimization type to "Minimum Volatility"
        self.initialization.prec_opt = "Minimum Volatility"

        # If allocations is True, save the allocation and return a DataFrame of allocations
        # The "x" key in the result object contains the optimal allocations that minimize volatility
        if record_optimized_allocation:
            asset_allocation = output["x"]
            self.initialization.asset_allocation = asset_allocation
            self.initialization.asset_allocation_dataframe = self._asset_allocation_dframe(asset_allocation)
            return self.initialization.asset_allocation_dataframe
        else:
            # If record_optimized_allocation is False, return the optimal allocations directly
            # These are the values that minimize the objective function
            return output["x"]   

    def mef_sharpe_maximisation(self, record_optimized_allocation=True):
        """
        Optimize the portfolio to maximize the Sharpe ratio.
        
        Parameters:
        - record_optimized_allocation: Boolean indicating whether to record the optimized allocation.
        
        Returns:
        - A DataFrame of asset allocations if record_optimized_allocation is True, otherwise the optimal allocations.
        """

        # Validate that recording the optimized asset allocations is a boolean
        if not isinstance(record_optimized_allocation, bool):
            raise ValueError("The variable record_optimized_allocation should be a boolean value.")

        # Extract average revenue and dispersion matrix values for optimization
        avg_revenue = self.initialization.avg_revenue.values
        disp_matrix = self.initialization.disp_matrix.values
        risk_free_ROR = self.initialization.risk_free_ROR

        # Extract other parameters required for optimization
        initial_guess = self.initialization.initial_guess
        method = self.initialization.method
        limits = self.initialization.limits
        constraints = self.initialization.constraints

        # Define the parameters for the optimization function
        param_opt = (avg_revenue, disp_matrix, risk_free_ROR)

        # Perform the optimization using SciPy's minimize function
        # The objective function is calculate_inverse_sharpe_ratio
        # The "x" in the output will contain the values that maximize the Sharpe ratio
        output = optimize.minimize(
            mef_minimisation.calculate_inverse_sharpe_ratio,
            args=param_opt,
            x0=initial_guess,
            method=method,
            bounds=limits,
            constraints=constraints,
        )

        # Update the preceding optimization type to "Maximum Sharpe Ratio"
        self.initialization.prec_opt = "Maximum Sharpe Ratio"

        # If record_optimized_allocation is True, save the allocation and return a DataFrame of allocations
        # The "x" key in the result object contains the optimal allocation that maximize the Sharpe ratio
        if record_optimized_allocation:
            asset_allocation = output["x"]
            self.initialization.asset_allocation = asset_allocation
            self.initialization.asset_allocation_dataframe = self._asset_allocation_dframe(asset_allocation)
            return self.initialization.asset_allocation_dataframe
        else:
            # If record_optimized_allocation is False, return the optimal allocation directly
            # These are the values that maximize the Sharpe ratio
            return output["x"]

    def mef_return(self, target_return, record_optimized_allocation=True):
        """
        Optimize the portfolio to achieve a specified target return while minimizing volatility.
        
        Parameters:
        - target_return: The desired return for the portfolio.
        - record_optimized_allocation: Boolean indicating whether to record the optimized allocation.
        
        Returns:
        - A DataFrame of asset allocations if record_optimized_allocation is True, otherwise the optimal allocations.
        """
            
        if not isinstance(target_return, (int, float)):
            raise ValueError("target_return is required as an integer or float.")
        if not isinstance(record_optimized_allocation, bool):
            raise ValueError("record_optimized_allocation is required as a boolean.")

        # Extract average revenue and dispersion matrix values for optimization
        avg_revenue = self.initialization.avg_revenue.values
        disp_matrix = self.initialization.disp_matrix.values

        # Extract other parameters required for optimization
        initial_guess = self.initialization.initial_guess
        method = self.initialization.method
        limits = self.initialization.limits

        # Define the parameters for the optimization function
        param_opt = (avg_revenue, disp_matrix)

        # Define the constraints for the optimization
        def constraint_func(x):
            return numpy.sum(x) - 1

        def return_constraint(x):
            return mef_minimisation.calculate_annualised_return(x, avg_revenue, disp_matrix) - target_return

        constraints = [{"type": "eq", "fun": constraint_func}, {"type": "eq", "fun": return_constraint}]

        # Perform the optimization using SciPy's minimize function
        # The objective function is calculate_annualized_volatility
        # The "x" in the output will contain the values that minimize the volatility
        output = optimize.minimize(
            mef_minimisation.calculate_annualized_volatility,
            args=param_opt,
            x0=initial_guess,
            method=method,
            bounds=limits,
            constraints=constraints,
        )

        # Update the preceding optimization type to "Efficient Return"
        self.initialization.prec_opt = "Efficient Return"

        # If record_optimized_allocation is True, save the allocation and return a DataFrame of allocation
        # The "x" key in the result object contains the optimal allocation that minimize volatility
        if record_optimized_allocation:
            asset_allocation = output["x"]
            self.initialization.asset_allocation = asset_allocation
            self.initialization.asset_allocation_dataframe = self._asset_allocation_dframe(asset_allocation)
            return self.initialization.asset_allocation_dataframe
        else:
            # If record_optimized_allocation is False, return the optimal allocation directly
            # These are the values that minimize the objective function
            return output["x"]
        
    def mef_volatility(self, target_volatility):
        """
        Optimize the portfolio to achieve a specified target volatility while maximizing the Sharpe ratio.
        
        Parameters:
        - target_volatility: The desired volatility for the portfolio.
        
        Returns:
        - A DataFrame of asset allocations.
        """  

        if not isinstance(target_volatility, (int, float)):
            raise ValueError("target_volatility is requried as an integer or float.")

        # Extract average revenue and dispersion matrix values for optimization
        avg_revenue = self.initialization.avg_revenue.values
        disp_matrix = self.initialization.disp_matrix.values

        # Extract other parameters required for optimization
        initial_guess = self.initialization.initial_guess
        method = self.initialization.method
        limits = self.initialization.limits

        # Define the parameters for the optimization function
        param_opt = (avg_revenue, disp_matrix, self.initialization.risk_free_ROR)

        # Define the constraints for the optimization
        def constraint_func(x):
            return numpy.sum(x) - 1

        def volatility_constraint(vx):
            return mef_minimisation.calculate_annualized_volatility(vx, avg_revenue, disp_matrix) - target_volatility

        constraints = [{"type": "eq", "fun": constraint_func}, {"type": "eq", "fun": volatility_constraint}]

        # Perform the optimization using SciPy's minimize function
        # The objective function is calculate_inverse_sharpe_ratio
        # The "x" in the output will contain the values that maximize the Sharpe ratio
        output = optimize.minimize(
            mef_minimisation.calculate_inverse_sharpe_ratio,
            args=param_opt,
            x0=initial_guess,
            method=method,
            bounds=limits,
            constraints=constraints,
        )

        # Update the preceding optimization type to "Efficient Volatility"
        self.initialization.prec_opt = "Efficient Volatility"

        # Record the allocation and return a DataFrame of allocations
        # The "x" key in the result object contains the optimal allocations that maximize the Sharpe ratio
        asset_allocation = output["x"]
        self.initialization.asset_allocation = asset_allocation
        self.initialization.asset_allocation_dataframe = self._asset_allocation_dframe(asset_allocation)
        return self.initialization.asset_allocation_dataframe
    
    def evaluate_mef(self, targets=None):
        """
        Evaluate the Markowitz Efficient Frontier (MEF) based on provided or generated targets.
        
        Parameters:
        - targets: List or numpy array of return targets. If not provided, they will be generated.
        
        Returns:
        - A numpy array of optimal MEF points.
        """
        # Validate and generate targets if not provided
        targets = self._validate_and_generate_targets(targets)
        
        # Calculate the efficient frontier based on the targets
        optimal_mef_points = self._calculate_efficient_frontier(targets)
        
        # Convert the efficient frontier list to a numpy array and save it
        self.initialization.optimal_mef_points = numpy.array(optimal_mef_points)
        return self.initialization.optimal_mef_points

    def _validate_and_generate_targets(self, targets):
        """
        Validate the provided targets or generate them if not provided.
        
        Parameters:
        - targets: List or numpy array of return targets.
        
        Returns:
        - Validated or generated targets.
        """
        # Check if targets are provided and are either a list or numpy array
        if targets is not None and not isinstance(targets, (list, numpy.ndarray)):
            raise ValueError("targets is requried as a list or numpy.ndarray")

        # If targets are not provided, generate them based on average revenue and trading days
        if targets is None:
            min_return = self.initialization.avg_revenue.min() * self.initialization.regular_trading_days
            max_return = self.initialization.avg_revenue.max() * self.initialization.regular_trading_days
            targets = numpy.linspace(round(min_return, 2), round(max_return, 2), 500)

        return targets

    def _calculate_efficient_frontier(self, targets):
        """
        Calculate the efficient frontier based on provided targets.
        
        Parameters:
        - targets: List or numpy array of return targets.
        
        Returns:
        - A list of optimal MEF points.
        """
        optimal_mef_points = [] # Optimised Markowitz Efficient Frontier points
        
        # Iterate through the targets, calculating the efficient return for each target
        for target in targets:
            asset_allocation = self.mef_return(target, record_optimized_allocation=False)
            
            # Calculate the annualized volatility for the given allocations
            annualized_return, annualized_volatility, sharpe_ratio = calculate_annualisation_of_measures(
                asset_allocation, self.initialization.avg_revenue, self.initialization.disp_matrix, regular_trading_days=self.initialization.regular_trading_days
            )
            
            # Append the annualized volatility and target to the efficient frontier list
            optimal_mef_points.append([annualized_volatility, target])

        return optimal_mef_points

class EfficientFrontierVisualization:
    """
    Class for visualizing the Markowitz Efficient Frontier.
    """
    def __init__(self, initialization, optimization):
        self.initialization = initialization
        self.optimization = optimization

    def _asset_allocation_dframe(self):
        """
        Retrieve the asset allocation dataframe from the initialization.
        
        Returns:
        - A DataFrame of asset allocations.
        """
        return self.initialization.asset_allocation_dataframe

    def plot_optimal_mef_points(self):
        """
        Plot the optimal MEF points on a graph.
        """
        # Check if the efficient frontier has been calculated; if not, calculate it
        if self.initialization.optimal_mef_points is None:
            self.optimization.evaluate_mef()

        # Extract the volatility and annualized return values from the efficient frontier
        annualised_volatility = self.initialization.optimal_mef_points[:, 0]
        annualized_return = self.initialization.optimal_mef_points[:, 1]

        # Call the private method to plot the efficient frontier with the extracted values
        self._plot_style_optimal_mef_points(annualised_volatility, annualized_return)

    def _plot_style_optimal_mef_points(self, volatility, annualized_return):
        """
        Style and plot the efficient frontier based on provided volatility and return values.
        
        Parameters:
        - volatility: List or numpy array of annualized volatility values.
        - annualized_return: List or numpy array of forecast return values.
        """
        # Plot the efficient frontier line using the given volatility and return values
        pylab.plot(volatility, annualized_return, linestyle='--', color='black', lw=1.5, label="Efficient Frontier")

        # Set the title and axis labels with specific font sizes
        pylab.title("Portfolio Optimisation: Markowitz Efficient Frontier", fontsize=16)
        pylab.xlabel("Annualised Volatility", fontsize=12)
        pylab.ylabel("Forecast Return", fontsize=12)

        # Add a grid to the plot for better readability
        pylab.grid(True, linestyle='--', alpha=0.5)

        # Add a legend to the plot with a frame, positioned in the upper left
        pylab.legend(frameon=True, loc='upper left')

        # Set the size of the plot (optional, can be adjusted as needed)
        pylab.gcf().set_size_inches(10, 6)

    def plot_vol_and_sharpe_optimal(self):
        """
        Plot the optimal portfolios for minimum volatility and maximum Sharpe ratio.
        """
        # Calculate the allocations and magnitudes for the minimum volatility portfolio
        optimal_minimal_volatility_allocation = self.optimization.mef_volatility_minimisation(record_optimized_allocation=False)
        optimal_minimal_volatility = self._mef_optimal_ordering(optimal_minimal_volatility_allocation)

        # Calculate the allocations and magnitudes for the maximum Sharpe ratio portfolio
        optimal_maximal_sharpe_allocation = self.optimization.mef_sharpe_maximisation(record_optimized_allocation=False)
        optimal_maximal_sharpe = self._mef_optimal_ordering(optimal_maximal_sharpe_allocation)

        # Plot the optimal portfolios using specific styles
        self._mef_optimal_style(optimal_minimal_volatility, "indigo", "x", "Minimum MEF Volatility") # X Marker
        self._mef_optimal_style(optimal_maximal_sharpe, "r", "x", "Maximum MEF Sharpe Ratio") # X marker

        # Add a legend to the plot with a frame
        pylab.legend(frameon=True, loc='upper left')

        # Add a grid for better readability
        pylab.grid(True, linestyle='--', alpha=0.5)

        # Optionally, change the size of the plot
        pylab.gcf().set_size_inches(10, 6)

    def _mef_optimal_ordering(self, asset_allocation):
        """
        Calculate and order the annualized volatility and return for a given asset allocation.
        
        Parameters:
        - asset_allocation: Allocations for assets.
        
        Returns:
        - Ordered list of magnitudes [annualized return, annualized volatility].
        """
        magnitudes = list(calculate_annualisation_of_measures(
            asset_allocation, self.initialization.avg_revenue, self.initialization.disp_matrix, regular_trading_days=self.initialization.regular_trading_days
        ))[0:2]
        magnitudes.reverse() # Reverse the order of the values
        return magnitudes

    def _mef_optimal_style(self, magnitudes, color, marker, label):
        """
        Plot a portfolio point on the graph with specific style attributes.
        
        Parameters:
        - magnitudes: List of [annualized return, annualized volatility].
        - color: Color of the point.
        - marker: Marker style.
        - label: Label for the point.
        """
        pylab.scatter(magnitudes[0], magnitudes[1], marker=marker, color=color, s=110, label=label)

    def mef_metrics(self, show_details=False):
        """
        Calculate and optionally display the metrics for the Markowitz Efficient Frontier.
        
        Parameters:
        - show_details: Boolean indicating if the metrics should be printed.
        
        Returns:
        - Tuple of (annualised_return, annualised_volatility, sharpe_ratio).
        """
        # Validate the input and the state of the object
        self._validate_mef_metrics_request(show_details)

        # Calculate the portfolio properties
        annualised_return, annualised_volatility, sharpe_ratio = self._calculate_mef_metrics()

        # Print the properties if show_details is True
        if show_details:
            self._print_mef_metrics(annualised_return, annualised_volatility, sharpe_ratio)

        # Return the calculated properties
        return (annualised_return, annualised_volatility, sharpe_ratio)

    def _validate_mef_metrics_request(self, show_details):
        """
        Validate the request for MEF metrics.
        
        Parameters:
        - show_details: Boolean indicating if the metrics should be printed.
        """
        # Check if show_details is a boolean
        if not isinstance(show_details, bool):
            raise ValueError("show_details is required as a boolean.")

        # Check if an optimization has been performed
        if self.initialization.asset_allocation is None:
            raise ValueError("Perform an optimisation first.")

    def _calculate_mef_metrics(self):
        """
        Calculate the metrics for the Markowitz Efficient Frontier.
        
        Returns:
        - Tuple of (annualised_return, annualsied_volatility, sharpe_ratio).
        """
        return calculate_annualisation_of_measures(
            self.initialization.asset_allocation,
            self.initialization.avg_revenue,
            self.initialization.disp_matrix,
            risk_free_ROR=self.initialization.risk_free_ROR,
            regular_trading_days=self.initialization.regular_trading_days,
        )

    def _print_mef_metrics(self, annualised_return, annualised_volatility, sharpe_ratio):
        """
        Print the metrics for the Markowitz Efficient Frontier in a formatted manner.
        
        Parameters:
        - annualised_return: The annualised return of the portfolio.
        - annualised_volatility: The annualised volatility of the portfolio.
        - sharpe_ratio: The Sharpe ratio of the portfolio.
        """
        # Create a formatted string with the portfolio properties
        stats = "=" * 50  # Use '=' as separator
        stats += f"\nOptimised Portfolio for {self.initialization.prec_opt}\n"
        stats += "-" * 50  # Use '-' to separate sections
        stats += f"\nEpoch/Trading Days: {self.initialization.regular_trading_days}"
        stats += f"\nRisk-Free Rate of Return: {self.initialization.risk_free_ROR:.2%}"  # Display as percentage
        stats += f"\nPredicted Annualised Return: {annualised_return:.3%}"  # Display as percentage
        stats += f"\nAnnualised Volatility: {annualised_volatility:.3%}"  # Display as percentage
        stats += f"\nSharpe Ratio: {sharpe_ratio:.4f}\n"
        stats += "-" * 50  # Use '-' to separate sections

        stats += "\nMost favourable Allocation:"
        # Get the asset allocation DataFrame
        asset_allocation_df = self._asset_allocation_dframe()
        # Transpose the DataFrame for better presentation
        transposed_allocation = asset_allocation_df.transpose()
        # Convert to string and add to the existing string with line breaks
        stats += f"\n{str(transposed_allocation)}\n"
        stats += "=" * 50  # Use '=' as a separator

        # Print the formatted string
        print(stats)

class EfficientFrontierMaster:
    """
    Master class that integrates the initialization, optimization, and visualization of the Efficient Frontier.
    """
    def __init__(self, avg_revenue, disp_matrix, risk_free_ROR=0.005427, regular_trading_days=252, method="SLSQP"):
        """
        Initialize the master class with the given parameters.
        """
        self.initialization = EfficientFrontierInitialization(avg_revenue, disp_matrix, risk_free_ROR, regular_trading_days, method)
        self.optimization = EfficientFrontierOptimization(self.initialization)
        self.visualization = EfficientFrontierVisualization(self.initialization, self.optimization)

    # Properties that delegate to the Initialization component
    
    @property
    def avg_revenue(self):
        return self.initialization.avg_revenue

    @property
    def disp_matrix(self):
        return self.initialization.disp_matrix

    @property
    def risk_free_ROR(self):
        return self.initialization.risk_free_ROR

    @property
    def regular_trading_days(self):
        return self.initialization.regular_trading_days

    @property
    def method(self):
        return self.initialization.method

    @property
    def symbol_stocks(self):
        return self.initialization.symbol_stocks

    @property
    def portfolio_size(self):
        return self.initialization.portfolio_size

    @property
    def asset_allocation(self):
        return self.initialization.asset_allocation

    @property
    def asset_allocation_dataframe(self):
        return self.initialization.asset_allocation_dataframe

    @property
    def optimal_mef_points(self):
        return self.initialization.optimal_mef_points
    
    # Methods that delegate to the Optimization component

    def mef_volatility_minimisation(self, record_optimized_allocation=True):
        return self.optimization.mef_volatility_minimisation(record_optimized_allocation)

    def mef_sharpe_maximisation(self, record_optimized_allocation=True):
        return self.optimization.mef_sharpe_maximisation(record_optimized_allocation)

    def mef_return(self, target, record_optimized_allocation=True):
        return self.optimization.mef_return(target, record_optimized_allocation)

    def mef_volatility(self, target):
        return self.optimization.mef_volatility(target)

    def mef_evaluate_mef(self, targets=None):
        return self.optimization.evaluate_mef(targets)
    
    # Methods that delegate to the Visualization component

    def mef_plot_optimal_mef_points(self):
        return self.visualization.plot_optimal_mef_points()

    def mef_plot_vol_and_sharpe_optimal(self):
        return self.visualization.plot_vol_and_sharpe_optimal()

    def mef_metrics(self, show_details=False):
        return self.visualization.mef_metrics(show_details)
