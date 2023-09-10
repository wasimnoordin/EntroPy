import pytest
import pandas

from src.investment import Investment

# This module contains tests for the Investment class

# Fixtures setup for the tests
@pytest.fixture
def mock_asset_price_history():
    """Provide a mock asset price history for testing."""
    return pandas.Series([100, 101, 102, 103, 104, 105], index=pandas.date_range(start="2023-01-01", periods=6))

@pytest.fixture
def investment_instance(mock_asset_price_history):
    """Provide a sample Investment instance for testing."""
    return Investment(mock_asset_price_history, "TestAsset", "Stock")

# Tests for the Investment class
def test_investment_initialization(mock_asset_price_history):
    """Ensure Investment instance is correctly initialized."""
    investment = Investment(mock_asset_price_history, "TestAsset", "Stock")
    # Check properties after initialization
    assert investment.investment_name == "TestAsset"
    assert investment.investment_category == "Stock"
    assert investment.asset_price_history.equals(mock_asset_price_history)

def test_calculate_investment_daily_return(investment_instance, mock_asset_price_history):
    """Ensure daily returns are computed correctly."""
    daily_returns = investment_instance.calculate_investment_daily_return().dropna()
    expected_returns = mock_asset_price_history.pct_change().dropna()
    assert daily_returns.equals(expected_returns)

def test_calculate_forecast_investment_return(investment_instance):
    """Ensure forecasted returns are computed correctly."""
    forecast_return = investment_instance.calculate_forecast_investment_return()
    daily_returns = investment_instance.calculate_investment_daily_return().dropna()
    expected_forecast_return = daily_returns.mean() * 252  # Annualized mean
    assert forecast_return == expected_forecast_return

def test_calculate_annualised_investment_volatility(investment_instance):
    """Ensure annualized volatility is computed correctly."""
    volatility = investment_instance.calculate_annualised_investment_volatility()
    daily_returns = investment_instance.calculate_investment_daily_return()
    expected_volatility = (daily_returns.std() * (252 ** 0.5))
    assert volatility == expected_volatility

def test_calculate_investment_skewness(investment_instance):
    """Ensure skewness is computed correctly."""
    skewness = investment_instance.calculate_investment_skewness()
    assert skewness == investment_instance.asset_price_history.skew()

def test_calculate_investment_kurtosis(investment_instance):
    """Ensure kurtosis is computed correctly."""
    kurtosis = investment_instance.calculate_investment_kurtosis()
    assert kurtosis == investment_instance.asset_price_history.kurt()

def test_display_attributes(investment_instance, capsys):
    """Ensure the display method prints expected attributes."""
    investment_instance.display_attributes()
    captured = capsys.readouterr()
    # Check if the attributes are present in the captured output
    assert "Forecast Return" in captured.out
    assert "Annualised Volatility" in captured.out
    assert "Investment Skew" in captured.out
    assert "Investment Kurtosis/Tailedness" in captured.out

def test_str_representation(investment_instance):
    """Ensure the string representation of Investment is as expected."""
    str_repr = str(investment_instance)
    # Check if the attributes are present in the string representation
    assert "TestAsset" in str_repr
    assert "Stock" in str_repr
    assert "Forecast Return" in str_repr
    assert "Annualised Volatility" in str_repr
