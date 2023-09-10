import pytest
import pandas
import numpy
import matplotlib.pyplot as pyplot

from src.markowitz_efficient_frontier import (
    EfficientFrontierInitialization,
    EfficientFrontierOptimization,
    EfficientFrontierVisualization,
    EfficientFrontierMaster,
)

# Switching the backend for matplotlib to "Agg" for better compatibility across systems
pyplot.switch_backend("Agg")

def mock_data():
    """
    Generates mock data for average revenue and dispersion matrix.
    Returns: Mocked average revenue and dispersion matrix.
    """

    avg_revenue = pandas.Series([0.05, 0.03, 0.04], index=["stockA", "stockB", "stockC"])
    disp_matrix = pandas.DataFrame({
        "stockA": [0.1, 0.02, 0.03],
        "stockB": [0.02, 0.1, 0.04],
        "stockC": [0.03, 0.04, 0.1]
    }, index=["stockA", "stockB", "stockC"])
    return avg_revenue, disp_matrix

# 1. Tests for EfficientFrontierInitialization

def test_initialization_valid_input():
    avg_revenue, disp_matrix = mock_data()
    ef_init = EfficientFrontierInitialization(avg_revenue, disp_matrix)
    assert ef_init.avg_revenue.equals(avg_revenue)
    assert ef_init.disp_matrix.equals(disp_matrix)
    assert ef_init.risk_free_ROR == 0.005427
    assert ef_init.regular_trading_days == 252
    assert ef_init.method == "SLSQP"
    assert ef_init.symbol_stocks == ["stockA", "stockB", "stockC"]
    assert ef_init.portfolio_size == 3

def test_initialization_invalid_input():
    avg_revenue, disp_matrix = mock_data()
    # Test invalid avg_revenue type
    with pytest.raises(ValueError):
        ef_init = EfficientFrontierInitialization("invalid_avg_revenue", disp_matrix)

    # Test invalid disp_matrix type
    with pytest.raises(ValueError):
        ef_init = EfficientFrontierInitialization(avg_revenue, "invalid_disp_matrix")

    # Test invalid risk_free_ROR
    with pytest.raises(ValueError):
        ef_init = EfficientFrontierInitialization(avg_revenue, disp_matrix, risk_free_ROR=-0.01)

    # Test invalid regular_trading_days
    with pytest.raises(ValueError):
        ef_init = EfficientFrontierInitialization(avg_revenue, disp_matrix, regular_trading_days=0)

    # Test invalid method
    with pytest.raises(ValueError):
        ef_init = EfficientFrontierInitialization(avg_revenue, disp_matrix, method="INVALID_METHOD")

# 2. Tests for EfficientFrontierOptimization

@pytest.fixture
def optimization_fixture():
    avg_revenue, disp_matrix = mock_data()
    ef_init = EfficientFrontierInitialization(avg_revenue, disp_matrix)
    return EfficientFrontierOptimization(ef_init)

def test_mef_volatility_minimisation(optimization_fixture):
    """
    Test the mef_volatility_minimisation method of the EfficientFrontierOptimization class.
    The test checks if this method correctly updates the associated attributes of the class.
    """
    optimization_fixture.mef_volatility_minimisation()
    # Assert that the 'prec_opt' attribute is updated to "Minimum Volatility"
    assert optimization_fixture.initialization.prec_opt == "Minimum Volatility"
    # Assert that the 'asset_allocation' attribute is an instance of numpy array
    assert isinstance(optimization_fixture.initialization.asset_allocation, numpy.ndarray)
    # Assert that the 'asset_allocation_dataframe' attribute is an instance of DataFrame
    assert isinstance(optimization_fixture.initialization.asset_allocation_dataframe, pandas.DataFrame)

def test_mef_sharpe_maximisation(optimization_fixture):
    """
    Test the mef_sharpe_maximisation method of the EfficientFrontierOptimization class.
    The test verifies if the 'prec_opt' attribute is updated correctly after the method is called.
    """
    optimization_fixture.mef_sharpe_maximisation()
    # Assert that the 'prec_opt' attribute is updated to "Maximum Sharpe Ratio"
    assert optimization_fixture.initialization.prec_opt == "Maximum Sharpe Ratio"

def test_mef_return(optimization_fixture):
    """
    Test the mef_return method of the EfficientFrontierOptimization class.
    The method is checked for correct behavior with a given target return value.
    """
    target_return_value = 0.05
    allocation = optimization_fixture.mef_return(target_return_value)
    # Assert that the 'prec_opt' attribute is updated to "Efficient Return"
    assert optimization_fixture.initialization.prec_opt == "Efficient Return"
    # Assert that the method returns a DataFrame
    assert isinstance(allocation, pandas.DataFrame)

def test_mef_volatility(optimization_fixture):
    """
    Test the mef_volatility method of the EfficientFrontierOptimization class.
    The method is checked for correct behavior with a given target volatility value.
    """
    target_volatility_value = 0.1
    allocation_df = optimization_fixture.mef_volatility(target_volatility_value)
    # Assert that the 'prec_opt' attribute is updated to "Efficient Volatility"
    assert optimization_fixture.initialization.prec_opt == "Efficient Volatility"
    # Assert that the method returns a DataFrame
    assert isinstance(allocation_df, pandas.DataFrame)

def test_evaluate_mef(optimization_fixture):
    """
    Test the evaluate_mef method of the EfficientFrontierOptimization class.
    This test checks if the method correctly calculates and returns the optimal points for given targets.
    """
    targets = [0.05, 0.1, 0.15]
    optimal_points = optimization_fixture.evaluate_mef(targets)
    # Assert that the method returns a numpy array
    assert isinstance(optimal_points, numpy.ndarray)
    # Assert that the returned array has two columns representing volatility and return
    assert optimal_points.shape[1] == 2  

# 3. Tests for EfficientFrontierVisualization

@pytest.fixture
def visualization_fixture():
    """
    Pytest fixture that sets up and returns an instance of the EfficientFrontierVisualization class.
    This fixture is used to provide a consistent setup for tests that require visualization.
    """
    # Generate mock data for revenue and dispersion matrix
    avg_revenue, disp_matrix = mock_data()
    
    # Initialize the EfficientFrontierInitialization class with the mock data
    ef_init = EfficientFrontierInitialization(avg_revenue, disp_matrix)
    
    # Initialize the EfficientFrontierOptimization class with the previously initialized 'ef_init'
    ef_opt = EfficientFrontierOptimization(ef_init)
    
    # Return an instance of the EfficientFrontierVisualization class using the initialized objects
    return EfficientFrontierVisualization(ef_init, ef_opt)

def test_mef_metrics(visualization_fixture):
    """
    Test the mef_metrics method of the EfficientFrontierVisualization class.
    The test checks if this method returns the correct metrics and their data types.
    """
    # Apply the Sharpe maximization method on the optimization object of the visualization_fixture
    visualization_fixture.optimization.mef_sharpe_maximisation()  # Choose another optimization method if needed
    
    # Get the metrics using the mef_metrics method
    metrics = visualization_fixture.mef_metrics()
    
    # Assert that the metrics list contains three items (annualised return, annualised volatility, and Sharpe ratio)
    assert len(metrics) == 3  
    
    # Unpack the metrics for individual assertions
    annualised_return, annualised_volatility, sharpe_ratio = metrics
    
    # Assert that each metric is a float type
    assert isinstance(annualised_return, float)
    assert isinstance(annualised_volatility, float)
    assert isinstance(sharpe_ratio, float)

def test_mef_plot_optimal_mef_points(visualization_fixture):
    # This method mainly produces a plot, call it check for exceptions
    visualization_fixture.plot_optimal_mef_points()

def test_plot_vol_and_sharpe_optimal(visualization_fixture):
    # This method mainly produces a plot, call it to check for exceptions
    visualization_fixture.plot_vol_and_sharpe_optimal()

# 4. Tests for EfficientFrontierMaster

@pytest.fixture
def master_fixture():
    """
    Pytest fixture to initialize and return an instance of EfficientFrontierMaster.
    """
    avg_revenue, disp_matrix = mock_data()
    return EfficientFrontierMaster(avg_revenue, disp_matrix)

def test_master_properties(master_fixture):
    assert isinstance(master_fixture.avg_revenue, pandas.Series)
    assert isinstance(master_fixture.disp_matrix, pandas.DataFrame)
    assert master_fixture.risk_free_ROR == 0.005427

@pytest.fixture
def master_fixture():
    """
    Pytest fixture to set up and return an instance of EfficientFrontierMaster.
    This fixture uses mock data for revenue and dispersion matrix to initialize the object.
    """
    avg_revenue, disp_matrix = mock_data()
    return EfficientFrontierMaster(avg_revenue, disp_matrix)

def test_master_properties(master_fixture):
    """
    Test the properties of the EfficientFrontierMaster instance.
    This function checks if the attributes of the instance are correctly set and have the right data types.
    """
    # Ensure the 'avg_revenue' attribute is a pandas Series
    assert isinstance(master_fixture.avg_revenue, pandas.Series)
    
    # Ensure the 'disp_matrix' attribute is a pandas DataFrame
    assert isinstance(master_fixture.disp_matrix, pandas.DataFrame)
    
    # Check if the risk-free rate of return attribute is correctly set
    assert master_fixture.risk_free_ROR == 0.005427

def test_master_methods(master_fixture):
    """
    Test various methods of the EfficientFrontierMaster class.
    This function checks if each method returns the correct type and, in some cases, the correct values.
    """
    
    # Test if 'mef_volatility_minimisation' method returns a DataFrame
    allocation_df = master_fixture.mef_volatility_minimisation()
    assert isinstance(allocation_df, pandas.DataFrame)

    # Test if 'mef_sharpe_maximisation' method returns a DataFrame
    allocation_df = master_fixture.mef_sharpe_maximisation()
    assert isinstance(allocation_df, pandas.DataFrame)

    # Test if 'mef_return' method returns a DataFrame for a given target return value
    target_return_value = 0.05
    allocation = master_fixture.mef_return(target_return_value, record_optimized_allocation=True)
    assert isinstance(allocation, pandas.DataFrame)

    # Test if 'mef_volatility' method returns a DataFrame for a given target volatility value
    target_volatility_value = 0.1
    allocation_df = master_fixture.mef_volatility(target_volatility_value)
    assert isinstance(allocation_df, pandas.DataFrame)

    # Test if 'mef_evaluate_mef' method returns an array with correct shape for given targets
    targets = [0.05, 0.1, 0.15]
    optimal_points = master_fixture.mef_evaluate_mef(targets)
    assert isinstance(optimal_points, numpy.ndarray)
    assert optimal_points.shape[1] == 2  # Expecting 2 columns: volatility and return

    # Test 'mef_plot_optimal_mef_points' method. It primarily produces a plot, so just call it to check for exceptions
    master_fixture.mef_plot_optimal_mef_points()

    # Test 'mef_plot_vol_and_sharpe_optimal' method. Similarly, call to check for exceptions
    master_fixture.mef_plot_vol_and_sharpe_optimal()

    # Test 'mef_metrics' method to ensure it returns correct metrics and types
    metrics = master_fixture.mef_metrics()
    assert len(metrics) == 3
    annualised_return, annualised_volatility, sharpe_ratio = metrics
    assert isinstance(annualised_return, float)
    assert isinstance(annualised_volatility, float)
    assert isinstance(sharpe_ratio, float)

