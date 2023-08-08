import unittest
import pytest
import pandas as pd
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
from unittest.mock import patch

from pytest import param
from finquant.efficient_frontier import EfficientFrontier  # Import the Initialization class
from finquant.minimise_fun import (portfolio_volatility, negative_sharpe_ratio, portfolio_return)
import finquant.minimise_fun as min_fun
from finquant.quants import annualised_portfolio_quantities

import src.ef_minimisation
from src.markowitz_efficient_frontier import (
    EfficientFrontierInitialization, 
    EfficientFrontierOptimization,
    EfficientFrontierVisualization,
    EfficientFrontierMaster)

class TestInitializationClass(unittest.TestCase):

    def setUp(self):
        print("Setting up sample mean returns and covariance matrix...")
        self.avg_revenue = pd.Series([0.12, 0.18], index=['stock1', 'stock2'])
        self.disp_matrix = pd.DataFrame({
            'stock1': [0.1, 0.03],
            'stock2': [0.03, 0.12]
        }, index=['stock1', 'stock2'])

    def test_valid_initialization(self):
        print("Testing valid initialization of the EfficientFrontierInitialization class...")
        ef_init = EfficientFrontierInitialization(self.avg_revenue, self.disp_matrix)
        
        print("Checking mean returns...")
        self.assertEqual(ef_init.avg_revenue.tolist(), self.avg_revenue.tolist())
        print("Mean returns match expected values.")

        print("Checking covariance matrix...")
        self.assertEqual(ef_init.disp_matrix.values.tolist(), self.disp_matrix.values.tolist())
        print("Covariance matrix matches expected values.")

        print("Checking risk-free rate...")
        self.assertEqual(ef_init.risk_free_ROR, 0.005427)
        print("Risk-free rate matches expected value.")

        print("Checking trading days...")
        self.assertEqual(ef_init.regular_trading_days, 252)
        print("Trading days matches expected value.")

        print("Checking optimization method...")
        self.assertEqual(ef_init.method, "SLSQP")
        print("Optimization method matches expected value.")

        print("Checking bounds...")
        self.assertEqual(ef_init.limits, ((0, 1), (0, 1)))  # Based on the number of stocks
        print("Bounds match expected values.")

        print("Checking initial weights...")
        self.assertEqual(ef_init.initial_guess.tolist(), [0.5, 0.5])  # Equal weights for two stocks
        print("Initial weights match expected values.")

        print("Checking constraints by applying to a sample input...")
        sample_weights = np.array([0.5, 0.5])
        constraint_result = ef_init.constraints['fun'](sample_weights)
        self.assertEqual(constraint_result, 0)  # The constraint should be satisfied
        print("Constraint is satisfied for the given sample input.")

        print("All checks passed for valid initialization.\n")


    def test_invalid_avg_revenue(self):
        print("\nTesting initialization with invalid mean returns (not a pandas.Series)...")
        with self.assertRaises(ValueError):
            EfficientFrontierInitialization([0.12, 0.18], self.disp_matrix)
        print("Initialization with invalid mean returns raised ValueError as expected.\n")

    def test_invalid_disp_matrix(self):
        print("\nTesting initialization with invalid covariance matrix (not a pandas.DataFrame)...")
        with self.assertRaises(ValueError):
            EfficientFrontierInitialization(self.avg_revenue, [[0.1, 0.03], [0.03, 0.12]])
        print("Initialization with invalid covariance matrix raised ValueError as expected.\n")

    def test_unsupported_method(self):
        print("\nTesting initialization with unsupported optimization method...")
        with self.assertRaises(ValueError):
            EfficientFrontierInitialization(self.avg_revenue, self.disp_matrix, method="unsupported_method")
        print("Initialization with unsupported method raised ValueError as expected.\n")
    
    def test_empty_avg_revenue(self):
        print("\nTesting initialization with empty mean returns...")
        with self.assertRaises(ValueError):
            EfficientFrontierInitialization(pd.Series([], index=[]), self.disp_matrix)
        print("Initialization with empty mean returns raised ValueError as expected.\n")

    def test_mismatched_avg_revenue_and_disp_matrix(self):
        print("\nTesting initialization with mismatched mean returns and covariance matrix...")
        avg_revenue_mismatched = pd.Series([0.12], index=['stock1'])
        with self.assertRaises(ValueError):
            EfficientFrontierInitialization(avg_revenue_mismatched, self.disp_matrix)
        print("Initialization with mismatched mean returns and covariance matrix raised ValueError as expected.\n")

    def test_negative_risk_free_ROR(self):
        print("\nTesting initialization with negative risk-free rate...")
        with self.assertRaises(ValueError):
            EfficientFrontierInitialization(self.avg_revenue, self.disp_matrix, risk_free_ROR=-0.01)
        print("Initialization with negative risk-free rate raised ValueError as expected.\n")

    def test_invalid_regular_trading_days(self):
        print("\nTesting initialization with non-integer trading days...")
        with self.assertRaises(ValueError):
            EfficientFrontierInitialization(self.avg_revenue, self.disp_matrix, regular_trading_days="daily")
        print("Initialization with non-integer trading days raised ValueError as expected.\n")

class TestEfficientFrontierOptimization(unittest.TestCase):

    def setUp(self):
        avg_revenue = pd.Series([0.12, 0.18], index=['stock1', 'stock2'])
        disp_matrix = pd.DataFrame({
            'stock1': [0.1, 0.03],
            'stock2': [0.03, 0.12]
        }, index=['stock1', 'stock2'])
        self.initialization = EfficientFrontierInitialization(avg_revenue, disp_matrix)
        self.ef_optimization = EfficientFrontierOptimization(self.initialization)

    def test_minimum_volatility(self):
        result = self.ef_optimization.mef_volatility_minimisation()
        self.assertIsInstance(result, pd.DataFrame)
        # Assuming specific expected weights for minimum volatility
        expected_weights = pd.DataFrame([0.6, 0.4], index=['stock1', 'stock2'], columns=["Allocation"])
        pd.testing.assert_frame_equal(result, expected_weights)

"__________________________-"

####################################################
# for now, the corresponding module in finquant is #
# tested through portfolio                         #
####################################################


mean_returns = pd.Series([0.12, 0.18], index=['stock1', 'stock2'])
cov_matrix = pd.DataFrame({
    'stock1': [0.1, 0.03],
    'stock2': [0.03, 0.12]
}, index=['stock1', 'stock2'])

class TestInitializationClass(unittest.TestCase):

    def setUp(self):
        # Sample mean returns and covariance matrix
        self.mean_returns = pd.Series([0.12, 0.18], index=['stock1', 'stock2'])
        self.cov_matrix = pd.DataFrame({
            'stock1': [0.1, 0.03],
            'stock2': [0.03, 0.12]
        }, index=['stock1', 'stock2'])

    def test_valid_initialization(self):
        ef_init = EfficientFrontier(self.mean_returns, self.cov_matrix)
        self.assertEqual(ef_init.mean_returns.tolist(), self.mean_returns.tolist())
        self.assertEqual(ef_init.cov_matrix.values.tolist(), self.cov_matrix.values.tolist())
        self.assertEqual(ef_init.risk_free_rate, 0.005)
        self.assertEqual(ef_init.freq, 252)
        self.assertEqual(ef_init.method, "SLSQP")
        self.assertEqual(ef_init.bounds, ((0, 1), (0, 1)))  # Based on the number of stocks
        self.assertEqual(ef_init.x0.tolist(), [0.5, 0.5])  # Equal weights for two stocks
        
        # Test the constraint by applying it to a sample input
        sample_weights = np.array([0.5, 0.5])
        constraint_result = ef_init.constraints['fun'](sample_weights)
        self.assertEqual(constraint_result, 0)  # The constraint should be satisfied


    def test_invalid_mean_returns(self):
        with self.assertRaises(ValueError):
            EfficientFrontier([0.12, 0.18], self.cov_matrix)

    def test_invalid_cov_matrix(self):
        with self.assertRaises(ValueError):
            EfficientFrontier(self.mean_returns, [[0.1, 0.03], [0.03, 0.12]])

    def test_unsupported_method(self):
        with self.assertRaises(ValueError):
            EfficientFrontier(self.mean_returns, self.cov_matrix, method="unsupported_method")

class TestEfficientFrontierOptimization(unittest.TestCase):

    def setUp(self):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.ef_optimization = EfficientFrontier(mean_returns, cov_matrix)
            
    def test_minimum_volatility(self):
        result = self.ef_optimization.minimum_volatility()
        weights = result.values.flatten()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 1))
        self.assertAlmostEqual(result.sum().values[0], 1.0)
        # Check if the result has the minimum volatility among random portfolios
        min_volatility = portfolio_volatility(weights, mean_returns.values, cov_matrix.values)
        for _ in range(100):
            random_weights = np.random.dirichlet(np.ones(2), size=1).flatten()
            random_volatility = portfolio_volatility(random_weights, mean_returns.values, cov_matrix.values)
            self.assertGreaterEqual(random_volatility, min_volatility)
    
    def test_maximum_sharpe_ratio(self):
        # Perform the maximum Sharpe ratio optimization
        result = self.ef_optimization.maximum_sharpe_ratio()
        optimized_weights = result.values.flatten()

        # Ensure that the weights sum to 1 (total allocation)
        self.assertAlmostEqual(np.sum(optimized_weights), 1)

        # Ensure that the weights are non-negative
        self.assertTrue((optimized_weights >= 0).all())

        # Ensure that the expected return is valid
        expected_return = np.sum(mean_returns * optimized_weights)
        self.assertGreaterEqual(expected_return, 0.0)

        # Ensure that the Sharpe ratio is non-negative
        sharpe_ratio = expected_return / self.ef_optimization.risk_free_rate
        self.assertGreaterEqual(sharpe_ratio, 0.0)

        # Test with zero returns
        zero_returns = pd.Series([0.0, 0.0], index=['stock1', 'stock2'])
        ef_zero = EfficientFrontier(zero_returns, cov_matrix)
        result = ef_zero.maximum_sharpe_ratio()
        self.assertAlmostEqual(result['Allocation'].sum(), 1.0, places=6)

        # Stress test with a larger number of assets
        num_assets = 100
        np.random.seed(42)  # For reproducibility
        stress_mean_returns = pd.Series(np.random.rand(num_assets))
        stress_cov_matrix = pd.DataFrame(np.random.rand(num_assets, num_assets))
        stress_cov_matrix = (stress_cov_matrix + stress_cov_matrix.T) / 2  # Make it symmetric
        np.fill_diagonal(stress_cov_matrix.values, 1)  # Fill diagonal with 1s to make it a valid covariance matrix

        # Create an EfficientFrontier object
        ef_stress = EfficientFrontier(stress_mean_returns, stress_cov_matrix)

        # Perform the maximum Sharpe ratio optimization
        result_stress = ef_stress.maximum_sharpe_ratio()
        optimized_weights_stress = result_stress.values.flatten()

        # Ensure that the weights sum to 1 (total allocation)
        self.assertAlmostEqual(np.sum(optimized_weights_stress), 1, delta=1e-2)

        # Ensure that the weights are non-negative
        self.assertTrue((optimized_weights_stress >= 0).all())

        # Ensure that the expected return is valid
        expected_return_stress = np.sum(stress_mean_returns * optimized_weights_stress)
        self.assertGreaterEqual(expected_return_stress, 0.0)

        # Ensure that the Sharpe ratio is non-negative
        sharpe_ratio_stress = expected_return_stress / ef_stress.risk_free_rate
        self.assertGreaterEqual(sharpe_ratio_stress, 0.0)

        # Sensitivity test with a small change in mean returns
        small_change = 0.01
        altered_returns = mean_returns + small_change
        ef_original = EfficientFrontier(mean_returns, cov_matrix)
        ef_altered = EfficientFrontier(altered_returns, cov_matrix)

        result_original = ef_original.maximum_sharpe_ratio()
        result_altered = ef_altered.maximum_sharpe_ratio()

        # Check that the optimized weights are not exactly the same
        self.assertNotEqual(result_original.values.flatten().tolist(), result_altered.values.flatten().tolist())

        # Boundary test with minimum and maximum mean returns
        min_returns = pd.Series([0.0, 0.0], index=['stock1', 'stock2'])  # Minimum possible returns
        max_returns = pd.Series([1.0, 1.0], index=['stock1', 'stock2'])  # Maximum possible returns

        ef_min = EfficientFrontier(min_returns, cov_matrix)
        ef_max = EfficientFrontier(max_returns, cov_matrix)

        result_min = ef_min.maximum_sharpe_ratio()
        result_max = ef_max.maximum_sharpe_ratio()

        # Check that the optimized weights sum to 1 for both minimum and maximum returns
        self.assertAlmostEqual(result_min['Allocation'].sum(), 1.0, places=6)
        self.assertAlmostEqual(result_max['Allocation'].sum(), 1.0, places=6)

  
    def test_efficient_return(self):
        # Define mean returns and covariance matrix for a hypothetical set of stocks
        mean_returns = pd.Series([0.05, 0.08, 0.12])
        cov_matrix = pd.DataFrame([
            [0.0009, 0.0004, 0.0002],
            [0.0004, 0.0016, 0.0007],
            [0.0002, 0.0007, 0.0025]
        ])

        # Create an EfficientFrontier object
        ef = EfficientFrontier(mean_returns, cov_matrix)
        ef.freq = 1  # Set the frequency to 1 since we are dealing with annual returns

       # Define a broader range of annualized target returns
        target_returns = np.linspace(
            min(mean_returns),  # Note: Do not multiply with ef.freq here
            max(mean_returns) * 2,  # Note: Do not multiply with ef.freq here, using a factor of 2
            100
        )

        for target_return in target_returns:
            try:
                # Call the efficient_return method
                df_weights = ef.efficient_return(target_return)

                # Check that the result is a DataFrame with the correct shape
                assert isinstance(df_weights, pd.DataFrame)
                assert df_weights.shape == (3, 1)

                # Check that the weights sum to 1
                assert np.isclose(df_weights.sum().values[0], 1)

                # Convert the target return to annual return and apply ef.freq for scaling
                annual_target_return = target_return * ef.freq

                # Check that the portfolio's expected return is close to the target
                portfolio_return = min_fun.portfolio_return(df_weights.values.flatten(), ef.mean_returns, ef.cov_matrix)

                # Compare the relative difference with the specified tolerance using np.isclose()
                assert np.isclose(portfolio_return, annual_target_return, rtol=1e-4, atol=0)

            except AssertionError as e:
                print(f"Assertion Error: {e}")
                print(f"Target Return: {target_return}")
                print(f"Annual Target Return: {annual_target_return}")
                print(f"Portfolio Return: {portfolio_return}")
                break

    def test_efficient_volatility(self):
        # Define mean returns and covariance matrix for a hypothetical set of stocks
        mean_returns = pd.Series([0.05, 0.08, 0.12])
        cov_matrix = pd.DataFrame([
            [0.0009, 0.0004, 0.0002],
            [0.0004, 0.0016, 0.0007],
            [0.0002, 0.0007, 0.0025]
        ])

        # Create an EfficientFrontier object
        ef = EfficientFrontier(mean_returns, cov_matrix)
        ef.freq = 1  # Set the frequency to 1 since we are dealing with annual returns

        # Define a range of target volatilities
        target_volatilities = np.linspace(
            min(cov_matrix.values.diagonal())**0.5,
            max(cov_matrix.values.diagonal())**0.5 * 2,
            100
        )

        for target_volatility in target_volatilities:
            try:
                # Call the efficient_volatility method
                df_weights = ef.efficient_volatility(target_volatility)

                # Check that the result is a DataFrame with the correct shape
                assert isinstance(df_weights, pd.DataFrame)
                assert df_weights.shape == (3, 1)

                # Check that the weights sum to 1
                assert np.isclose(df_weights.sum().values[0], 1)

                # Check that the portfolio's volatility is close to the target
                portfolio_volatility = min_fun.portfolio_volatility(df_weights.values.flatten(), ef.mean_returns, ef.cov_matrix)

                # Compare the relative difference with the specified tolerance using np.isclose()
                assert np.isclose(portfolio_volatility, target_volatility, rtol=1e-4, atol=0)

            except AssertionError as e:
                print(f"Assertion Error: {e}")
                print(f"Target Volatility: {target_volatility}")
                print(f"Portfolio Volatility: {portfolio_volatility}")
                break

    def test_efficient_volatility_performance(self):

        # Define a larger set of mean returns and covariance matrix
        num_assets = 50
        np.random.seed(42)  # For reproducibility
        mean_returns = pd.Series(np.random.rand(num_assets))
        cov_matrix = pd.DataFrame(np.random.rand(num_assets, num_assets))
        cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Make it symmetric
        np.fill_diagonal(cov_matrix.values, 1)  # Fill diagonal with 1s to make it a valid covariance matrix

        # Create an EfficientFrontier object
        ef = EfficientFrontier(mean_returns, cov_matrix)
        ef.freq = 1  # Set the frequency to 1 since we are dealing with annual returns

        # Define a range of target volatilities
        target_volatilities = np.linspace(
            min(cov_matrix.values.diagonal())**0.5,
            max(cov_matrix.values.diagonal())**0.5 * 2,
            3  # Reduced number of points for performance testing
        )

        # Measure the start time
        start_time = time.time()

        for target_volatility in target_volatilities:
            # Call the efficient_volatility method
            df_weights = ef.efficient_volatility(target_volatility)

            # You can include the same checks as in the previous test here if needed

        # Measure the end time
        end_time = time.time()

        # Check that the execution time is within an acceptable limit
        execution_time = end_time - start_time
        acceptable_time_limit = 5  # You can adjust this value based on your requirements
        self.assertLess(execution_time, acceptable_time_limit, f"Execution time {execution_time} exceeded the acceptable limit")
    
    def test_efficient_frontier_method(self):

        mean_returns = pd.Series([0.05, 0.08, 0.12])
        cov_matrix = pd.DataFrame([
            [0.0009, 0.0004, 0.0002],
            [0.0004, 0.0016, 0.0007],
            [0.0002, 0.0007, 0.0025]
        ])

        ef = EfficientFrontier(mean_returns, cov_matrix)
        efrontier = ef.efficient_frontier()

        # Check that the result is a numpy array with the correct shape
        assert isinstance(efrontier, np.ndarray)
        assert efrontier.shape[1] == 2

        # Check that the volatilities are non-negative
        assert all(efrontier[:, 0] >= 0)

        # Test with custom targets
        targets = np.linspace(0.05, 0.12, 5)
        efrontier_custom = ef.efficient_frontier(targets=targets)

        # Check that the result is a numpy array with the correct shape
        assert isinstance(efrontier_custom, np.ndarray)
        assert efrontier_custom.shape[1] == 2

        # Test with invalid targets (not a list or numpy array)
        with pytest.raises(ValueError):
            ef.efficient_frontier(targets="invalid_targets")

        # Boundary case: Test with zero covariance matrix
        zero_cov_matrix = pd.DataFrame([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        ef_zero_cov = EfficientFrontier(mean_returns, zero_cov_matrix)
        efrontier_zero_cov = ef_zero_cov.efficient_frontier()

        # Check that the result is a numpy array with the correct shape
        assert isinstance(efrontier_zero_cov, np.ndarray)
        assert efrontier_zero_cov.shape[1] == 2

        # Check that the volatilities are zero
        assert all(efrontier_zero_cov[:, 0] == 0)

        # Randomized Testing
        for _ in range(4):  # Run 4 randomized tests
            num_stocks = np.random.randint(1, 11)  # Random number of stocks between 1 and 50
            random_mean = pd.Series(np.random.rand(num_stocks))
            random_cov = pd.DataFrame(np.random.rand(num_stocks, num_stocks))
            random_cov = (random_cov + random_cov.T) / 2
            np.fill_diagonal(random_cov.values, 1)

            ef_random = EfficientFrontier(random_mean, random_cov)
            efrontier_random = ef_random.efficient_frontier()

            # Check that the result is a numpy array with the correct shape
            assert isinstance(efrontier_random, np.ndarray)
            assert efrontier_random.shape[1] == 2

            # Check that the volatilities are non-negative
            assert all(efrontier_random[:, 0] >= 0)
    
class TestPlotEFrontier(unittest.TestCase):
   
    def setUp(self):
        
        # Create example data for testing
        np.random.seed(42)
        num_stocks = 5
        num_days = 100
        returns = np.random.rand(num_days, num_stocks) / 100
        cov_matrix = pd.DataFrame(returns).cov()
        mean_returns = pd.Series(returns.mean(axis=0))
        self.ef = EfficientFrontier(mean_returns, cov_matrix, risk_free_rate=0.01, freq=252)

    def test_plot_clearing(self):
        # Clear the current figure before plotting
        plt.gcf().clear()

        # Plot the efficient frontier
        self.ef.plot_efrontier()

        # Check if the plot contains the efficient frontier line
        lines = plt.gca().lines
        self.assertGreaterEqual(len(lines), 1)

    def test_input_data_integrity(self):
        # Test if the input data has the correct format and size
        self.assertEqual(len(self.ef.mean_returns), len(self.ef.cov_matrix))

    def test_plot_limits(self):
        # Run the efficient_frontier method before plotting
        self.ef.efficient_frontier()

        # Get the minimum and maximum values of the efficient frontier
        min_volatility = min(self.ef.efrontier[:, 0])
        max_volatility = max(self.ef.efrontier[:, 0])
        min_return = min(self.ef.efrontier[:, 1])
        max_return = max(self.ef.efrontier[:, 1])

        # Set the plot limits manually (you can adjust these limits based on your data)
        plt.xlim(min_volatility - 0.01, max_volatility + 0.01)
        plt.ylim(min_return - 0.01, max_return + 0.01)

        # Plot the efficient frontier
        self.ef.plot_efrontier()

        # Check if the plot has the correct limits
        self.assertAlmostEqual(plt.gca().get_xlim()[0], min_volatility - 0.01)
        self.assertAlmostEqual(plt.gca().get_xlim()[1], max_volatility + 0.01)
        self.assertAlmostEqual(plt.gca().get_ylim()[0], min_return - 0.01)
        self.assertAlmostEqual(plt.gca().get_ylim()[1], max_return + 0.01)

    def custom_plot_efrontier(self, line_style="--", line_color="blue"):
            # Customize the plot_efrontier method to include line_style and line_color arguments
            # Clear the current figure before plotting
            plt.gcf().clear()

            # Get the efficient frontier data
            self.ef.efficient_frontier()
            plt.plot(self.ef.efrontier[:, 0], self.ef.efrontier[:, 1], linestyle=line_style, color=line_color)

            # Check if the plot contains the efficient frontier line
            lines = plt.gca().lines
            self.assertGreaterEqual(len(lines), 1)

    def test_custom_colors_and_styles(self):

        # Clear the current figure before plotting
        plt.gcf().clear()

        # Create sample data for testing
        mean_returns = pd.Series([0.05, 0.1, 0.15, 0.2, 0.25])
        cov_matrix = pd.DataFrame([
            [0.1, 0.05, 0.02, 0.04, 0.01],
            [0.05, 0.12, 0.01, 0.03, 0.02],
            [0.02, 0.01, 0.11, 0.01, 0.03],
            [0.04, 0.03, 0.01, 0.15, 0.07],
            [0.01, 0.02, 0.03, 0.07, 0.1]
        ])

        # Create the EfficientFrontier object
        ef = EfficientFrontier(mean_returns, cov_matrix, risk_free_rate=0.01, freq=252)

        # Plot the efficient frontier with the default colors and style
        ef.plot_efrontier()

        # Check if the plot contains the efficient frontier line
        lines = plt.gca().lines
        self.assertGreaterEqual(len(lines), 1)

    def test_no_data(self):
        # Test when there are no valid data points (all NaNs)
        mean_returns = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
        cov_matrix = pd.DataFrame([[np.nan] * 5] * 5)
        ef_with_nan = EfficientFrontier(mean_returns, cov_matrix, risk_free_rate=0.01, freq=252)

        # Clear the current figure before plotting
        plt.gcf().clear()

        # Plot the efficient frontier with no data
        ef_with_nan.plot_efrontier()

        # Check if the plot contains exactly one line (the empty line for the axes)
        axes = plt.gcf().get_axes()
        for ax in axes:
            self.assertEqual(len(ax.get_lines()), 1, "Plot should contain exactly one line")

if __name__ == '__main__':
    unittest.main()

 