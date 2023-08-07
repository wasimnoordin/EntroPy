

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
        self.mark_eff_frontier = None
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

    def minimum_volatility(self, record_optimized_allocation=True):
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
            self.initialization.asset_allocation_dataframe = self._dataframe_weights(asset_allocation)
            return self.initialization.asset_allocation_dataframe
        else:
            # If record_optimized_allocation is False, return the optimal allocations directly
            # These are the values that minimize the objective function
            return output["x"]   

    def maximum_sharpe_ratio(self, record_optimized_allocation=True):
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
            self.initialization.asset_allocation_dataframe = self._dataframe_weights(asset_allocation)
            return self.initialization.asset_allocation_dataframe
        else:
            # If record_optimized_allocation is False, return the optimal allocation directly
            # These are the values that maximize the Sharpe ratio
            return output["x"]

    def efficient_return(self, target_return, record_optimized_allocation=True):
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
            allocation = output["x"]
            self.initialization.asset_allocation = allocation
            self.initialization.asset_allocation_dataframe = self._dataframe_weights(allocation)
            return self.initialization.asset_allocation_dataframe
        else:
            # If record_optimized_allocation is False, return the optimal allocation directly
            # These are the values that minimize the objective function
            return output["x"]
        
    def efficient_volatility(self, target_volatility):
        
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

        def volatility_constraint(x):
            return ef_minimisation.calculate_annualized_volatility(x, avg_revenue, disp_matrix) - target_volatility

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

        # Save the weights and return a DataFrame of weights
        # The "x" key in the result object contains the optimal weights that maximize the Sharpe ratio
        allocation = output["x"]
        self.initialization.asset_allocation = allocation
        self.initialization.asset_allocation_dataframe = self._dataframe_weights(allocation)
        return self.initialization.asset_allocation_dataframe
    
    """______________"""

   # Compare function calls to see if each function is actually calling the correct