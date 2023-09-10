import numpy
import pandas
from matplotlib import pyplot

from src.measures_common import calculate_annualisation_of_measures

class MonteCarloSimulation:
    """A class for executing Monte Carlo Simulations (mcs)."""

    def __init__(self, mcs_iterations=None):
        """
        Parameters:
        -- mc_iterations: int or None, number of iterations for the
                 Monte Carlo simulation. 
        """
        self.mcs_iterations = mcs_iterations if mcs_iterations is not None else 5000
    
    def mcs_simulation(self, run_instance, **params):
        """
        Parameters:
        -- run_instance: A user-defined function that will be called at each 
                            iteration of the Monte Carlo simulation.
        -- params: syntax is used to pass a variable number of keyword arguments
                     to a function. 

        Returns:
        mc_run_output: Comprises of values returned from run_instance after n iterations.
        """

        # Use  list comprehension to call the function and store the results
        mcs_run_output = list(map(lambda _: run_instance(**params), range(self.mcs_iterations)))

        # Converting the list to a NumPy array with a specified data type
        return numpy.asarray(mcs_run_output, dtype=object)

class MonteCarloMethodology(MonteCarloSimulation):
    """A tool designed to execute Monte Carlo simulations to discover optimal financial portfolios.
    
    Parameters:
    - asset_revenue: Pandas DataFrame, comprises historical asset revenue. 
    - mcs_iteration: number of iterations for the Monte Carlo simulation.
    - risk_free_ROR: Float, risk-free rate of return. Obtained as 5.427% on 28/07/23 
                                from https://www.marketwatch.com/investing/bond/tmubmusd03m?countrycode=bx
    - regular_trading_days (optional): Numeric, average number of regular trading days in a year. 
                                Setpoint: 252 days. Taken from: https://en.wikipedia.org/wiki/Trading_day
    - seed_allocation: List / numpy.ndarray, proportions of original portfolio splits. 
                        Primary purpose is to designate the start-point of the portfolio in subsequent optimisation graphs.

    Returns:
    - xyz_optimal: Pandas DataFrame, minimise volatility and maximise Sharpe ratio. 
    """

    def __init__(
        self,
        asset_revenue,
        mcs_iterations=999,
        risk_free_ROR=0.005427,
        regular_trading_days=252,
        seed_allocation=None,
    ):
        # Validate the input parameters
        self.validate_inputs(asset_revenue, mcs_iterations, risk_free_ROR, regular_trading_days, seed_allocation)
         # Call the constructor of the superclass (MonteCarloSimulation) with the specified number of iterations
        super().__init__(mcs_iterations=mcs_iterations)
         # Initialize the attributes of the class
        self.initialize_attributes(asset_revenue, mcs_iterations, risk_free_ROR, regular_trading_days, seed_allocation)

    @staticmethod
    def validate_inputs(asset_revenue, mcs_iterations, risk_free_ROR, regular_trading_days, seed_allocation):
         # Validate the types and values of the input parameters
        if seed_allocation is not None and not isinstance(seed_allocation, numpy.ndarray):
            raise ValueError("The 'seed_allocation' parameter must be provided as a numpy.ndarray.")
        if not isinstance(asset_revenue, pandas.DataFrame):
            raise ValueError("The 'asset_revenue' parameter must be a pandas DataFrame containing the asset revenue data.")
        if not isinstance(mcs_iterations, int):
            raise ValueError("The 'mcs_iterations' parameter must be an integer representing the number of Monte Carlo simulation iterations.")
        if not isinstance(risk_free_ROR, (int, float)):
            raise ValueError("The 'risk_free_ROR' parameter must be either an integer or a floating-point number, representing the risk-free rate.")
        if not isinstance(regular_trading_days, int):
            raise ValueError("The 'regular_trading_days' parameter must be an integer representing the number of regular trading days considered in the analysis.")

    def initialize_attributes(self, asset_revenue, mcs_iterations, risk_free_ROR, regular_trading_days, seed_allocation):
        # Initialize the attributes of the class with the given parameters
        self.asset_revenue = asset_revenue
        self.mcs_iterations = mcs_iterations
        self.risk_free_ROR = risk_free_ROR
        self.regular_trading_days = regular_trading_days
        self.seed_allocation = seed_allocation if seed_allocation is not None else numpy.array([])

        # Set the portfolio metrics (e.g., number of assets, average revenue, dispersion matrix)
        self.set_portfolio_metrics()

    def set_portfolio_metrics(self):
         # Determine the number of assets in the portfolio
        self.asset_count = self.asset_revenue.shape[1]

        # Calculate the mean revenue for each asset
        self.avg_revenue = self.asset_revenue.mean()
        # Calculate the dispersion matrix of the asset returns
        self.disp_matrix = self.asset_revenue.cov()

        # Initialize attributes for storing the results of the optimization
        self.proportion_data_frame = None
        self.product_data_frame = None
        self.proportion_optimal = None
        self.product_optimal = None

    def _generate_uniform_allocations(self):
        """ --- INTERNAL / PRIVATE METHOD ---

        Calculates uniformly distributed random allocations for portfolio stocks alongside 
        their respective annualised returns, volatilities, and Sharpe Ratios.

        Parameters:
        -- asset_count: int, number of stocks in a specified portfolio

        Returns:
        -- pfl_holdings: Tuple of each asset proportion and list of annualised returns, volatilities and Sharpe
        """

        # Generate random allocations using a uniform distribution
        random_proportion = numpy.random.uniform(size=self.asset_count)
        # Normalize the allocations so they sum to 1
        proportion_of_investment = random_proportion / numpy.linalg.norm(random_proportion, ord=1)

        # Compute portfolio values using the external function
        pfl_holdings = calculate_annualisation_of_measures(
            proportion_of_investment, self.avg_revenue, self.disp_matrix, self.risk_free_ROR, self.regular_trading_days
        )

        return (proportion_of_investment, numpy.array(pfl_holdings))
    
    def _generate_portfolio_diversity(self):
        """ --- INTERNAL / PRIVATE METHOD ---
        
        Generates a list of diversified portfolios, alongside respective annualised return, volatility, and Sharpe Ratios. 

        Returns:
        -- proportion_data_frame: Pandas Dataframe, allocation of each randomly generated portfolio
        -- product_data_frame: Pandas Dataframe, return, volatility and Sharpe for each randomly generated portfolio
        """

        # Define a function to be used within the Monte Carlo simulation
        def execute_instance():
            return self._generate_uniform_allocations()

        # Run Monte Carlo simulation using the mcs_simulation method
        mcs_simulation_output = self.mcs_simulation(execute_instance)

        # Extract proportions and products from the Monte Carlo output
        proportion_list = [item[0] for item in mcs_simulation_output]
        product_list = [item[1] for item in mcs_simulation_output]

        # Create pandas DataFrames for proportions and results
        proportion_col = list(self.asset_revenue.columns)
        product_col = ["Annualised Return", "Volatility", "Sharpe Ratio"]
        proportion_data_frame = pandas.DataFrame(data=proportion_list, columns=proportion_col)
        product_data_frame = pandas.DataFrame(data=product_list, columns=product_col)

        return (proportion_data_frame, product_data_frame)

    def mcs_optimised_portfolio(self):
        """Portfolio Optimisation via actual execution of the Monte Carlo method.

        Parameters:
        -- self: Instance of the class where this method is executed. It contains attributes such as proportion_data_frame and product_data_frame.
        -- self._generate_portfolio_diversity(): A method within the same class, responsible for performing a Monte Carlo run to generate random portfolios and their corresponding results. 
                                                    It returns two pandas DataFrames, proportion_data_frame and product_data_frame, containing the proportions and product metrics of the generated portfolios.

        Returns:
        -- optimised_proportion: Pandas Dataframe, catalogs the optimal proportions for portfolios utilising portfolios with minimum volatility and maximum Sharpe.
        -- optimised_product: Pandas Dataframe, catalogs optimal product metrics (returns, volatility, sharpe) utilising portfolios with minimum volatility and maximum Sharpe.
        """

        # Perform Monte Carlo run to get random portfolios and their corresponding results
        proportion_data_frame, product_data_frame = self._generate_portfolio_diversity()

        # Check if the DataFrames are empty, and raise an error if no portfolios were generated
        if proportion_data_frame.empty or product_data_frame.empty:
            raise ValueError("No portfolios generated. Cannot proceed with optimization.")

        # "argmax/argmin" is acceptable if product_data_frame[Volatility/Sharpe Ratio] is a simple integer index starting at 0
        # Find the index of the portfolio with minimum volatility 
        volatility_minimisation = product_data_frame["Volatility"].values.argmin()
        # Find the index of the portfolio with maximum Sharpe Ratio
        sharpe_ratio_maximisation = product_data_frame["Sharpe Ratio"].values.argmax()

        # Catalogue the indices of the optimal portfolios in a dictionary
        optimal_indices = {
            "Minimum Volatility": volatility_minimisation,
            "Maximum Sharpe Ratio": sharpe_ratio_maximisation
        }

        # Extract the optimal proportions for portfolios and retain them in a DataFrame
        optimised_proportion = pandas.DataFrame(
            {key: proportion_data_frame.iloc[val] for key, val in optimal_indices.items()}
        ).T
        # Extract the optimal products (annualised return, volatility, Sharpe) and retain them in a DataFrame
        optimised_product = pandas.DataFrame(
            {key: product_data_frame.iloc[val] for key, val in optimal_indices.items()}
        ).T

        # Set instance variables to record the allocations and results of all portfolios, as well as the optimal ones
        self.proportion_data_frame = proportion_data_frame
        self.product_data_frame = product_data_frame
        self.optimised_proportion = optimised_proportion
        self.optimised_product = optimised_product

        # Return the optimal allocations and output
        return optimised_proportion, optimised_product  

    def mcs_visualisation(self): 
        """Outlines Monte Carlo Simulation visualisations with all portfolios and markers
          for min Volatility and max Sharpe Ratio.".
        """

        # Check that the necessary data is available
        if any(getattr(self, attr) is None for attr in ['product_data_frame', 'proportion_data_frame', 'optimised_proportion', 'optimised_product']):
            raise RuntimeError("Error: Plot is awaiting initial pass of Monte Carlo optimisation.")

        # Create a scatter plot colored by Sharpe Ratio
        pyplot.scatter(
            self.product_data_frame["Volatility"],
            self.product_data_frame["Annualised Return"],
            c=self.product_data_frame["Sharpe Ratio"],
            cmap="viridis",  # Colormap
            s=15,            # Point size
        )
        cbar = pyplot.colorbar()

        # Define the properties for the different types of portfolios
        portfolio_types = {
            "Minimum Volatility": {"color": "royalblue", "label": "Minimum Volatility"},
            "Maximum Sharpe Ratio": {"color": "hotpink", "label": "Maximum Sharpe Ratio"},
        }

        # Plot the portfolios
        for portfolio_type, properties in portfolio_types.items():
            pyplot.scatter(
                self.optimised_product.loc[portfolio_type]["Volatility"],
                self.optimised_product.loc[portfolio_type]["Annualised Return"],
                marker="*",       # Marker
                color=properties["color"],
                s=150,            # Size for the markers
                label=properties["label"],
            )

        # Plot the initial portfolio, if allocations were given
        if self.seed_allocation is not None:
            prelim_config = calculate_annualisation_of_measures(
                self.seed_allocation,
                self.avg_revenue,
                self.disp_matrix,
                self.risk_free_ROR,
                self.regular_trading_days,
            )
            pyplot.scatter(
                prelim_config[1],  # Volatility
                prelim_config[0],  # Annualised Return
                marker="1",         # Marker
                color="black",     # Color
                s=150,              # Size for the marker
                label="Initial Portfolio",
            )

        # Set the title and labels
        pyplot.title(f"Portfolio Optimization: Monte Carlo Simulation", fontsize = 16)
        pyplot.xlabel(f"Volatility", fontsize = 12)
        pyplot.ylabel(f"Annualised Return", fontsize = 12)
        cbar.ax.set_ylabel(f"Sharpe Ratio (Tradiing Days: {self.regular_trading_days})", rotation=90)
        pyplot.legend(loc='upper left')  # Legend location

        # # Show the plot
        # pyplot.show()

    def mcs_print_attributes(self):
        """Prints out the properties of the Monte Carlo Simulation."""

        # Check if the necessary data is available
        if self.optimised_product is None or self.optimised_proportion is None or self.regular_trading_days is None:
            raise ValueError("Error: Plot is awaiting initial pass of Monte Carlo optimisation.")

        # Define a dictionary to map optimization types to human-readable names
        optimized_parameters = {
            "Minimum Volatility": "Portfolio with Minimum Volatility",
            "Maximum Sharpe Ratio": "Portfolio with Maximum Sharpe Ratio",
        }

        # Function to print a single property
        def print_attribute(name, value, precision=4):
            print(f"{name.ljust(20)}: {value:.{precision}f}")

        # Iterate through the optimization types and print properties
        for optimized_parameter_type, description in optimized_parameters.items():
            print("=" * 50)
            print(description)
            print(f"\nTime period (days): {self.regular_trading_days}")
            print_attribute("\nAnnualised Return", self.optimised_product.loc[optimized_parameter_type]['Annualised Return'])
            print_attribute("Volatility", self.optimised_product.loc[optimized_parameter_type]['Volatility'])

            print("Optimized parameter type:", optimized_parameter_type)
            print("Optimized product DataFrame:\n", self.optimised_product)

            print_attribute("Sharpe Ratio", self.optimised_product.loc[optimized_parameter_type]['Sharpe Ratio'])
            print("\nOptimal allocations: \n")
            for asset, allocation in self.optimised_product.loc[optimized_parameter_type].items():
                print_attribute(asset, allocation)
            print("=" * 50)
            print()
