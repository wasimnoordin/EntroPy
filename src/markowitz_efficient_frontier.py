"""
This file defines a set of classes and methods for portfolio optimization using
the concept of the Efficient Frontier, a principle from Modern Portfolio Theory.

It includes classes for initializing the Efficient Frontier with various
constraints and methods, optimizing the portfolio based on an objective function
for different criteria (such as minimizing volatility or maximizing Sharpe ratio),
and visualizing the results through plots of the Efficient Frontier and 
portfolio metrics.

The code also provides validation and formatting functions to ensure proper
input and present the results in a user-friendly manner.

"""

import numpy
import pandas
import ef_minimisation 
from src.measures import calculate_annualisation_of_measures
from scipy import optimize
import matplotlib.pylab as pylab

# Compare optimisation against random search

class EfficientFrontierInitialization:

    def __init__(
        self, avg_revenue, disp_matrix, risk_free_ROR=0.005427, regular_trading_days=252, method="SLSQP"
    ):
        """
        Assigned method is SLSQP by default as it handles constraints, it is efficient, it is a gradient-based method, and enables quadratic approximation.
        
        Inspired by: 
        -- https://towardsdatascience.com/efficient-frontier-optimize-portfolio-with-scipy-57456428323e
        -- https://pub.towardsai.net/portfolio-management-using-python-portfolio-optimization-8a90dd2a21d
        
        """

        self.validate_inputs(avg_revenue, disp_matrix, risk_free_ROR, regular_trading_days, method)

        # instance variables
        self.avg_revenue = avg_revenue
        self.disp_matrix = disp_matrix
        self.risk_free_ROR = risk_free_ROR
        self.regular_trading_days = regular_trading_days
        self.method = method
        self.symbol_stocks = list(avg_revenue.index)
        self.portfolio_size = len(self.symbol_stocks)
        self.prec_opt = ""

        # set numerical parameters
        limit = (0, 1)
        self.limits = numpy.full((self.portfolio_size, 2), (0, 1))
        self.initial_guess = numpy.full(self.portfolio_size, 1.0 / self.portfolio_size)
        # Define a named function for the constraint
        def constraint_func(x):
            return numpy.sum(x) - 1
        # Set the constraint in the initialization
        self.constraints = {"type": "eq", "fun": constraint_func}

        # placeholder for optimised values/allocations
        self.optimal_mef_points = None
        self.asset_allocation = None
        self.asset_allocation_dataframe = None

    def validate_inputs(self, avg_revenue, disp_matrix, risk_free_ROR, regular_trading_days, method):
        supported_solvers = [
            "BFGS", "CG", "COBYLA", "dogleg", "L-BFGS-B", 
            "Nelder-Mead", "Newton-CG", "Powell", "SLSQP", "TNC",
            "trust-constr", "trust-exact", "trust-krylov", "trust-ncg"
        ]
        if not isinstance(avg_revenue, pandas.Series) or avg_revenue.empty:
            raise ValueError("avg_revenue is expected to be a non-empty pandas.Series.")
        if not isinstance(disp_matrix, pandas.DataFrame) or disp_matrix.empty:
            raise ValueError("disp_matrix is expected to be a non-empty pandas.DataFrame")
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


class EfficientFrontierOptimization:

    def __init__(self, initialization):
        self.initialization = initialization

    def mef_volatility_minimisation(self, record_optimized_allocation=True):
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
            ef_minimisation.calculate_annualized_volatility,
            param_opt=param_opt,
            initial_guess=initial_guess,
            method=method,
            limits=limits,
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
            ef_minimisation.calculate_inverse_sharpe_ratio,
            param_opt=param_opt,
            initial_guess=initial_guess,
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
        if not isinstance(target_return, (int, float)):
            raise ValueError("target_return is expected to be an integer or float.")
        if not isinstance(record_optimized_allocation, bool):
            raise ValueError("record_optimized_allocation is expected to be a boolean.")

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
            return ef_minimisation.calculate_annualised_return(x, avg_revenue, disp_matrix) - target_return

        constraints = [{"type": "eq", "fun": constraint_func}, {"type": "eq", "fun": return_constraint}]

        # Perform the optimization using SciPy's minimize function
        # The objective function is calculate_annualized_volatility
        # The "x" in the output will contain the values that minimize the volatility
        output = optimize.minimize(
            ef_minimisation.calculate_annualized_volatility,
            param_opt=param_opt,
            initial_guess=initial_guess,
            method=method,
            limits=limits,
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
        
        if not isinstance(target_volatility, (int, float)):
            raise ValueError("target_volatility is expected to be an integer or float.")

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
            return ef_minimisation.calculate_annualized_volatility(vx, avg_revenue, disp_matrix) - target_volatility

        constraints = [{"type": "eq", "fun": constraint_func}, {"type": "eq", "fun": volatility_constraint}]

        # Perform the optimization using SciPy's minimize function
        # The objective function is calculate_inverse_sharpe_ratio
        # The "x" in the output will contain the values that maximize the Sharpe ratio
        output = optimize.minimize(
            ef_minimisation.calculate_inverse_sharpe_ratio,
            param_opt=param_opt,
            initial_guess=initial_guess,
            method=method,
            limits=limits,
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
        # Validate and generate targets if not provided
        targets = self._validate_and_generate_targets(targets)
        # Calculate the efficient frontier based on the targets
        optimal_mef_points = self._calculate_efficient_frontier(targets)
        # Convert the efficient frontier list to a numpy array and save it
        self.initialization.optimal_mef_points = numpy.array(optimal_mef_points)
        return self.initialization.optimal_mef_points

    def _validate_and_generate_targets(self, targets):
        # Check if targets are provided and are either a list or numpy array
        if targets is not None and not isinstance(targets, (list, numpy.ndarray)):
            raise ValueError("targets is expected to be a list or numpy.ndarray")

        # If targets are not provided, generate them based on average revenue and trading days
        if targets is None:
            min_return = self.initialization.avg_revenue.min() * self.initialization.regular_trading_days
            max_return = self.initialization.avg_revenue.max() * self.initialization.regular_trading_days
            targets = numpy.linspace(round(min_return, 2), round(max_return, 2), 500)

        return targets

    def _calculate_efficient_frontier(self, targets):
        optimal_mef_points = [] # Optimised Markowitz Efficient Frontier points
        # Iterate through the targets, calculating the efficient return for each target
        for target in targets:
            asset_allocation = self.mef_return(target, record_optimized_allocation=False)
            # Calculate the annualized volatility for the given allocations
            annualized_volatility, _ = calculate_annualisation_of_measures(
                asset_allocation, self.initialization.avg_revenue, self.initialization.disp_matrix, regular_trading_days=self.initialization.regular_trading_days
            )
            # Append the annualized volatility and target to the efficient frontier list
            optimal_mef_points.append([annualized_volatility, target])

        return optimal_mef_points

class EfficientFrontierVisualization:

    def __init__(self, initialization, optimization):
        self.initialization = initialization
        self.optimization = optimization

    def plot_optimal_mef_points(self):
        # Check if the efficient frontier has been calculated; if not, calculate it
        if self.initialization.optimal_mef_points is None:
            self.evaluate_mef()

        # Extract the volatility and annualized return values from the efficient frontier
        annualised_volatility = self.initialization.optimal_mef_points[:, 0]
        annualized_return = self.initialization.optimal_mef_points[:, 1]

        # Call the private method to plot the efficient frontier with the extracted values
        self._plot_style_optimal_mef_points(annualised_volatility, annualized_return)

    def _plot_style_optimal_mef_points(self, volatility, annualized_return):
        # Plot the efficient frontier line using the given volatility and return values
        pylab.plot(volatility, annualized_return, linestyle='-', color='blue', lw=2, label="Efficient Frontier")

        # Set the title and axis labels with specific font sizes
        pylab.title("Markowitz Efficient Frontier", fontsize=16)
        pylab.xlabel("Annualised Volatility", fontsize=14)
        pylab.ylabel("Annualised Return", fontsize=14)

        # Add a grid to the plot for better readability
        pylab.grid(True, linestyle='--', alpha=0.5)

        # Add a legend to the plot with a frame, positioned in the upper left
        pylab.legend(frameon=True, loc='upper left')

        # Set the size of the plot (optional, can be adjusted as needed)
        pylab.gcf().set_size_inches(10, 6)

        # Display the plot
        pylab.show()

    def plot_vol_and_sharpe_optimal(self):
        # Calculate the allocations and magnitudes for the minimum volatility portfolio
        optimal_minimal_volatility_allocation = self.initialization.mef_volatility_minimisation(record_optimized_allocation=False)
        optimal_minimal_volatility = self._mef_optimal_ordering(optimal_minimal_volatility_allocation)

        # Calculate the allocations and magnitudes for the maximum Sharpe ratio portfolio
        optimal_maximal_sharpe_allocation = self.initialization.mef_sharpe_maximisation(record_optimized_allocation=False)
        optimal_maximal_sharpe = self._mef_optimal_ordering(optimal_maximal_sharpe_allocation)

        # Plot the optimal portfolios
        self._mef_optimal_style(optimal_minimal_volatility, "g", "o", "MEF Volatility Minimisation") # Green circle marker
        self._mef_optimal_style(optimal_maximal_sharpe, "r", "s", "MEF Sharpe Maximisation") # Red square marker

        # Add a legend to the plot with a frame
        pylab.legend(frameon=True, loc='upper left')

        # Add a grid for better readability
        pylab.grid(True, linestyle='--', alpha=0.5)

        # Optionally, change the size of the plot
        pylab.gcf().set_size_inches(10, 6)

        # Show the plot
        pylab.show()

    def _mef_optimal_ordering(self, asset_allocation):
        # Calculate the annualized volatility and return for the given allocation
        magnitudes = list(calculate_annualisation_of_measures(
            asset_allocation, self.initialization.avg_revenue, self.initialization.disp_matrix, regular_trading_days=self.initialization.regular_trading_days
        ))[0:2]
        magnitudes.reverse() # Reverse the order of the values
        return magnitudes

    def _mef_optimal_style(self, magnitudes, color, marker, label):
        # Plot a portfolio on the graph with the given values, color, marker, and label
        pylab.scatter(magnitudes[0], magnitudes[1], marker=marker, color=color, s=150, label=label)
    
    def _asset_allocation_dframe(self, allocation):
        # Check if allocation is a numpy.ndarray
        if not isinstance(allocation, numpy.ndarray):
            raise ValueError("allocation is expected to be a numpy.ndarray")

        # Check if the length of allocation array matches the number of asset designations
        if len(allocation) != len(self.initialization.symbol_stocks):
            raise ValueError("Length of allocation array must match the number of asset symbols")

        # Create a pandas DataFrame with the allocation, indexed by asset designations
        # and with a single column named "Allocation"
        return pandas.DataFrame(allocation, index=self.initialization.symbol_stocks, columns=["Allocation"])

    def mef_metrics(self, show_details=False):
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
        # Check if show_details is a boolean
        if not isinstance(show_details, bool):
            raise ValueError("show_details is expected to be a boolean.")

        # Check if an optimization has been performed
        if self.initialization.asset_allocation is None:
            raise ValueError("Perform an optimisation first.")

    def _calculate_mef_metrics(self):
        # Calculate and return the annualised return, annualsied volatility, and Sharpe ratio
        return calculate_annualisation_of_measures(
            self.initialization.asset_allocation,
            self.initialization.avg_revenue,
            self.initialization.disp_matrix,
            risk_free_ROR=self.initialization.risk_free_ROR,
            regular_trading_days=self.initialization.regular_trading_days,
        )

    def _print_mef_metrics(self, annualised_return, annualised_volatility, sharpe_ratio):
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
        asset_allocation_df = self._asset_allocation_dframe(self.initialization.asset_allocation_dataframe)
        # Transpose the DataFrame for better presentation
        transposed_allocation = asset_allocation_df.transpose()
        # Convert to string and add to the existing string with line breaks
        stats += f"\n{str(transposed_allocation)}\n"
        stats += "=" * 50  # Use '=' as a separator

        # Print the formatted string
        print(stats)

    
class EfficientFrontierMaster:
    def __init__(self, avg_revenue, disp_matrix, risk_free_ROR=0.005427, regular_trading_days=252, method="SLSQP"):
        self.initialization = EfficientFrontierInitialization(avg_revenue, disp_matrix, risk_free_ROR, regular_trading_days, method)
        self.optimization = EfficientFrontierOptimization(self.initialization)
        self.visualization = EfficientFrontierVisualization(self.initialization, self.optimization)

    def mef_volatility_minimisation(self, record_optimized_allocation=True):
        return self.optimization.mef_volatility_minimisation(record_optimized_allocation)

    def mef_sharpe_maximisation(self, record_optimized_allocation=True):
        return self.optimization.mef_sharpe_maximisation(record_optimized_allocation)

    def mef_return(self, target, record_optimized_allocation=True):
        return self.optimization.mef_return(target, record_optimized_allocation)

    def mef_volatility(self, target):
        return self.optimization.mef_volatility(target)

    def efficient_frontier(self, targets=None):
        return self.optimization.evaluate_mef(targets)

    def plot_optimal_mef_points(self):
        return self.visualization.plot_optimal_mef_points()

    def plot_vol_and_sharpe_optimal(self):
        return self.visualization.plot_vol_and_sharpe_optimal()

    def mef_metrics(self, show_details=False):
        return self.visualization.mef_metrics(show_details)
