import numpy
import pandas
import pytest
import matplotlib.pyplot as pyplot

from src.monte_carlo_simulation import MonteCarloMethodology

import pytest
import pandas
import numpy

pyplot.switch_backend("Agg")

# Sample data for testing
sample_asset_revenue = pandas.DataFrame({
    "Asset1": numpy.random.rand(10),
    "Asset2": numpy.random.rand(10)
})

# 1. Initialization Tests

def test_valid_initialization():
    """Test if the MonteCarloMethodology class initializes correctly with valid input."""
    mcm = MonteCarloMethodology(asset_revenue=sample_asset_revenue)
    assert isinstance(mcm, MonteCarloMethodology)

def test_invalid_seed_allocation_type():
    """Test initialization with invalid seed allocation type (list instead of numpy array)."""
    with pytest.raises(ValueError):
        MonteCarloMethodology(asset_revenue=sample_asset_revenue, seed_allocation=[0.5, 0.5])

def test_invalid_asset_revenue_type():
    """Test initialization with invalid asset revenue type (list of lists instead of DataFrame)."""
    with pytest.raises(ValueError):
        MonteCarloMethodology(asset_revenue=[[0.1, 0.2], [0.3, 0.4]])

def test_invalid_mcs_iterations_type():
    """Test initialization with invalid Monte Carlo Simulation iterations type (string instead of integer)."""
    with pytest.raises(ValueError):
        MonteCarloMethodology(asset_revenue=sample_asset_revenue, mcs_iterations="1000")

def test_invalid_risk_free_ROR_type():
    """Test initialization with invalid risk-free Rate Of Return type (string instead of float)."""
    with pytest.raises(ValueError):
        MonteCarloMethodology(asset_revenue=sample_asset_revenue, risk_free_ROR="0.005")

def test_invalid_regular_trading_days_type():
    """Test initialization with invalid regular trading days type (string instead of integer)."""
    with pytest.raises(ValueError):
        MonteCarloMethodology(asset_revenue=sample_asset_revenue, regular_trading_days="252")

def test_proper_initialization_of_attributes():
    """Test if the MonteCarloMethodology class initializes its attributes correctly."""
    mcm = MonteCarloMethodology(asset_revenue=sample_asset_revenue)
    # Ensure the asset revenue attribute is initialized correctly
    assert mcm.asset_revenue.equals(sample_asset_revenue)
    # Check default values for various attributes
    assert mcm.mcs_iterations == 999
    assert mcm.risk_free_ROR == 0.005427
    assert mcm.regular_trading_days == 252
    # Ensure the seed allocation attribute is initialized as an empty array
    assert mcm.seed_allocation.size == 0

# 2. Validation Tests

def test_valid_validate_inputs():
    """Test the validate_inputs method with valid inputs. It should not raise any exceptions."""
    MonteCarloMethodology.validate_inputs(sample_asset_revenue, 1000, 0.005, 252, numpy.array([0.5, 0.5]))

def test_invalid_validate_inputs_seed_allocation():
    """Test the validate_inputs method with invalid seed allocation type (list instead of numpy array)."""
    with pytest.raises(ValueError):
        MonteCarloMethodology.validate_inputs(sample_asset_revenue, 1000, 0.005, 252, [0.5, 0.5])

def test_invalid_validate_inputs_asset_revenue():
    """Test the validate_inputs method with invalid asset revenue type (list of lists instead of DataFrame)."""
    with pytest.raises(ValueError):
        MonteCarloMethodology.validate_inputs([[0.1, 0.2], [0.3, 0.4]], 1000, 0.005, 252, numpy.array([0.5, 0.5]))

def test_invalid_validate_inputs_mcs_iterations():
    """Test the validate_inputs method with invalid Monte Carlo Simulation iterations type (string instead of integer)."""
    with pytest.raises(ValueError):
        MonteCarloMethodology.validate_inputs(sample_asset_revenue, "1000", 0.005, 252, numpy.array([0.5, 0.5]))

def test_invalid_validate_inputs_risk_free_ROR():
    """Test the validate_inputs method with invalid risk-free Rate Of Return type (string instead of float)."""
    with pytest.raises(ValueError):
        MonteCarloMethodology.validate_inputs(sample_asset_revenue, 1000, "0.005", 252, numpy.array([0.5, 0.5]))

def test_invalid_validate_inputs_regular_trading_days():
    """Test the validate_inputs method with invalid regular trading days type (string instead of integer)."""
    with pytest.raises(ValueError):
        MonteCarloMethodology.validate_inputs(sample_asset_revenue, 1000, 0.005, "252", numpy.array([0.5, 0.5]))

# 3. Portfolio Metrics Tests

def test_set_portfolio_metrics():
    """Test if the set_portfolio_metrics method correctly sets the portfolio metrics attributes."""
    mcm = MonteCarloMethodology(asset_revenue=sample_asset_revenue)
    mcm.set_portfolio_metrics()
    
    # Ensure the asset count is correctly set
    assert mcm.asset_count == 2
    # Check if the average revenue is a pandas Series
    assert isinstance(mcm.avg_revenue, pandas.Series)
    # Check if the dispersion matrix is a pandas DataFrame
    assert isinstance(mcm.disp_matrix, pandas.DataFrame)

# 4. Uniform Allocations Tests

def test_generate_uniform_allocations():
    """Test if the _generate_uniform_allocations method correctly generates uniform allocations."""
    mcm = MonteCarloMethodology(asset_revenue=sample_asset_revenue)
    proportions, results = mcm._generate_uniform_allocations()
    
    # Ensure the proportions are correctly shaped
    assert proportions.shape[0] == 2
    # Check if the sum of proportions is close to 1
    assert numpy.isclose(numpy.sum(proportions), 1.0)
    # Ensure the results contain three metrics: Annualised Return, Volatility, and Sharpe Ratio
    assert len(results) == 3

# 5. Portfolio Diversity Tests

def test_generate_portfolio_diversity():
    """Test if the _generate_portfolio_diversity method correctly generates portfolio diversity data."""
    mcm = MonteCarloMethodology(asset_revenue=sample_asset_revenue)
    prop_df, prod_df = mcm._generate_portfolio_diversity()
    
    # Ensure the proportions DataFrame is correctly shaped
    assert isinstance(prop_df, pandas.DataFrame)
    # Ensure the product DataFrame is correctly shaped
    assert isinstance(prod_df, pandas.DataFrame)
    # Check if the proportions DataFrame has two columns
    assert prop_df.shape[1] == 2
    # Ensure the product DataFrame contains three metrics: Annualised Return, Volatility, and Sharpe Ratio
    assert prod_df.shape[1] == 3

# 6. Optimized Portfolio Tests

def test_mcs_optimised_portfolio():
    """Test if the mcs_optimised_portfolio method correctly generates optimized portfolio data."""
    mcm = MonteCarloMethodology(asset_revenue=sample_asset_revenue)
    optimised_proportion, optimised_product = mcm.mcs_optimised_portfolio()
    
    # Ensure the optimised proportions are stored in a DataFrame
    assert isinstance(optimised_proportion, pandas.DataFrame)
    # Ensure the optimised product metrics are stored in a DataFrame
    assert isinstance(optimised_product, pandas.DataFrame)
    # Check if the optimised proportions DataFrame has two columns
    assert optimised_proportion.shape[1] == 2
    # Ensure the optimised product DataFrame contains three metrics: Annualised Return, Volatility, and Sharpe Ratio
    assert optimised_product.shape[1] == 3

# 7. Visualization Tests

def test_mcs_visualisation():
    """Test if the mcs_visualisation method correctly visualizes the Monte Carlo simulation results."""
    # Provide seed_allocation to avoid the error
    seed_alloc = numpy.array([0.5, 0.5])
    
    mcm = MonteCarloMethodology(asset_revenue=sample_asset_revenue, seed_allocation=seed_alloc)
    mcm.mcs_optimised_portfolio()  # This method should be called before visualization
    # Ensure the visualization method runs without raising any exceptions
    try:
        mcm.mcs_visualisation()
        assert True
    except Exception as e:
        pytest.fail(f"Unexpected error raised: {e}")

# 8. Print Attributes Tests

def test_mcs_print_attributes():
    """Test if the mcs_print_attributes method correctly prints the attributes of the Monte Carlo simulation."""
    mcm = MonteCarloMethodology(asset_revenue=sample_asset_revenue)
    mcm.mcs_optimised_portfolio()  # This method should be called before printing attributes
    # Ensure the print attributes method runs without raising any exceptions
    try:
        mcm.mcs_print_attributes()
        assert True
    except Exception as e:
        pytest.fail(f"Unexpected error raised: {e}")
