"""
Test linear model implementation against R fixtures.

Validates that PyRegression produces identical results to R's lm()
within numerical tolerance.
"""

import pytest
import numpy as np
import json
from pathlib import Path

from pyregression import LinearModel


# Tolerance levels (based on double precision)
COEF_TOL = 1e-12      # Coefficient tolerance
RESID_TOL = 1e-12     # Residual tolerance  
STAT_TOL = 1e-10      # Statistics tolerance (R², etc.)


def load_fixture(name):
    """Load a test fixture from JSON."""
    fixture_path = Path(__file__).parent / "fixtures" / f"{name}.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


def test_simple_regression():
    """Test simple well-conditioned regression (100 obs, 3 predictors)."""
    fixture = load_fixture("simple_regression")
    
    X = np.array(fixture["X"])
    y = np.array(fixture["y"])
    
    # Fit with PyRegression
    model = LinearModel()
    result = model.fit(X, y)
    
    # Compare coefficients
    r_coef = np.array(fixture["coefficients"])
    np.testing.assert_allclose(
        result.coef, r_coef, rtol=COEF_TOL, atol=COEF_TOL,
        err_msg="Coefficients don't match R"
    )
    
    # Compare residuals
    r_resid = np.array(fixture["residuals"])
    np.testing.assert_allclose(
        result.residuals, r_resid, rtol=RESID_TOL, atol=RESID_TOL,
        err_msg="Residuals don't match R"
    )
    
    # Compare fitted values
    r_fitted = np.array(fixture["fitted_values"])
    np.testing.assert_allclose(
        result.fitted_values, r_fitted, rtol=RESID_TOL, atol=RESID_TOL,
        err_msg="Fitted values don't match R"
    )
    
    # Compare R²
    r_r2 = fixture["r_squared"]
    np.testing.assert_allclose(
        result.r_squared, r_r2, rtol=STAT_TOL, atol=STAT_TOL,
        err_msg="R² doesn't match R"
    )
    
    # Compare standard errors
    r_se = np.array(fixture["std_errors"])
    np.testing.assert_allclose(
        result.se, r_se, rtol=STAT_TOL, atol=STAT_TOL,
        err_msg="Standard errors don't match R"
    )
    
    # Compare rank
    assert result.rank == fixture["rank"], "Rank doesn't match R"
    
    # Compare df_residual
    assert result.df_residual == fixture["df_residual"], "DF residual doesn't match R"


def test_large_regression():
    """Test larger problem (500 obs, 10 predictors)."""
    fixture = load_fixture("large_regression")
    
    X = np.array(fixture["X"])
    y = np.array(fixture["y"])
    
    model = LinearModel()
    result = model.fit(X, y)
    
    # Compare key outputs
    r_coef = np.array(fixture["coefficients"])
    np.testing.assert_allclose(
        result.coef, r_coef, rtol=COEF_TOL, atol=COEF_TOL
    )
    
    r_resid = np.array(fixture["residuals"])
    np.testing.assert_allclose(
        result.residuals, r_resid, rtol=RESID_TOL, atol=RESID_TOL
    )
    
    r_r2 = fixture["r_squared"]
    np.testing.assert_allclose(
        result.r_squared, r_r2, rtol=STAT_TOL, atol=STAT_TOL
    )


def test_rank_deficient():
    """Test rank deficient case (perfect collinearity)."""
    fixture = load_fixture("rank_deficient")
    
    X = np.array(fixture["X"])
    y = np.array(fixture["y"])
    
    model = LinearModel()
    result = model.fit(X, y, singular_ok=True)
    
    # Check that rank is detected correctly
    assert result.rank == fixture["rank"], "Rank not detected correctly"
    assert result.rank < (X.shape[1] + 1), "Should be rank deficient"
    
    # Compare non-NA coefficients
    r_coef = np.array(fixture["coefficients"])
    
    # Both should have NAs in same positions
    py_na_mask = np.isnan(result.coef)
    r_na_mask = np.isnan(r_coef)
    np.testing.assert_array_equal(
        py_na_mask, r_na_mask,
        err_msg="NA pattern in coefficients doesn't match R"
    )
    
    # Compare non-NA coefficients
    non_na = ~py_na_mask
    np.testing.assert_allclose(
        result.coef[non_na], r_coef[non_na],
        rtol=COEF_TOL, atol=COEF_TOL,
        err_msg="Non-NA coefficients don't match R"
    )
    
    # Residuals should still match
    r_resid = np.array(fixture["residuals"])
    np.testing.assert_allclose(
        result.residuals, r_resid, rtol=RESID_TOL, atol=RESID_TOL
    )


def test_ill_conditioned():
    """Test ill-conditioned case (near collinearity)."""
    fixture = load_fixture("ill_conditioned")
    
    X = np.array(fixture["X"])
    y = np.array(fixture["y"])
    
    model = LinearModel()
    result = model.fit(X, y)
    
    # Should handle near-collinearity gracefully
    r_coef = np.array(fixture["coefficients"])
    
    # May have slightly larger errors due to conditioning
    # but should still be close
    np.testing.assert_allclose(
        result.coef, r_coef, rtol=1e-8, atol=1e-8,
        err_msg="Coefficients differ too much on ill-conditioned problem"
    )
    
    # Residuals should match well
    r_resid = np.array(fixture["residuals"])
    np.testing.assert_allclose(
        result.residuals, r_resid, rtol=RESID_TOL, atol=RESID_TOL
    )


def test_single_predictor():
    """Test simplest case (single predictor)."""
    fixture = load_fixture("single_predictor")
    
    X = np.array(fixture["X"])
    y = np.array(fixture["y"])
    
    model = LinearModel()
    result = model.fit(X, y)
    
    # Everything should match exactly
    r_coef = np.array(fixture["coefficients"])
    np.testing.assert_allclose(
        result.coef, r_coef, rtol=COEF_TOL, atol=COEF_TOL
    )
    
    r_resid = np.array(fixture["residuals"])
    np.testing.assert_allclose(
        result.residuals, r_resid, rtol=RESID_TOL, atol=RESID_TOL
    )
    
    r_r2 = fixture["r_squared"]
    np.testing.assert_allclose(
        result.r_squared, r_r2, rtol=STAT_TOL, atol=STAT_TOL
    )


def test_intercept_only():
    """Test edge case (no predictors, intercept only)."""
    fixture = load_fixture("intercept_only")
    
    # Empty X matrix
    X = np.array(fixture["X"]).reshape(-1, 0)  # n x 0 matrix
    y = np.array(fixture["y"])
    
    model = LinearModel()
    result = model.fit(X, y)
    
    # Should have only intercept
    r_coef = np.array(fixture["coefficients"])
    assert len(result.coef) == 1, "Should have only intercept"
    np.testing.assert_allclose(
        result.coef, r_coef, rtol=COEF_TOL, atol=COEF_TOL
    )
    
    # Fitted values should all be mean of y
    assert np.allclose(result.fitted_values, np.mean(y))
    
    # R² should be 0 (no predictors)
    assert result.r_squared == 0.0


def test_variance_covariance_matrix():
    """Test that variance-covariance matrix matches R."""
    fixture = load_fixture("simple_regression")
    
    X = np.array(fixture["X"])
    y = np.array(fixture["y"])
    
    model = LinearModel()
    result = model.fit(X, y)
    
    # Compare vcov matrix
    r_vcov = np.array(fixture["vcov"])
    np.testing.assert_allclose(
        result.vcov, r_vcov, rtol=STAT_TOL, atol=STAT_TOL,
        err_msg="Variance-covariance matrix doesn't match R"
    )


@pytest.mark.parametrize("fixture_name", [
    "simple_regression",
    "large_regression",
    "single_predictor",
])
def test_standard_errors_from_vcov(fixture_name):
    """Verify that SE = sqrt(diag(vcov))."""
    fixture = load_fixture(fixture_name)
    
    X = np.array(fixture["X"])
    y = np.array(fixture["y"])
    
    model = LinearModel()
    result = model.fit(X, y)
    
    # Standard errors should be sqrt of diagonal of vcov
    se_from_vcov = np.sqrt(np.diag(result.vcov))
    np.testing.assert_allclose(
        result.se, se_from_vcov, rtol=1e-14, atol=1e-14,
        err_msg="SE doesn't match sqrt(diag(vcov))"
    )


def test_summary_statistics():
    """Test that summary statistics are computed correctly."""
    fixture = load_fixture("simple_regression")
    
    X = np.array(fixture["X"])
    y = np.array(fixture["y"])
    
    model = LinearModel()
    result = model.fit(X, y)
    
    # Check adj_r_squared
    r_adj_r2 = fixture["adj_r_squared"]
    np.testing.assert_allclose(
        result.adj_r_squared, r_adj_r2, rtol=STAT_TOL, atol=STAT_TOL
    )
    
    # Check sigma (residual standard error)
    r_sigma = fixture["sigma"]
    # Compute from our residuals
    n = len(y)
    rss = np.sum(result.residuals ** 2)
    our_sigma = np.sqrt(rss / result.df_residual)
    
    np.testing.assert_allclose(
        our_sigma, r_sigma, rtol=STAT_TOL, atol=STAT_TOL,
        err_msg="Residual standard error doesn't match R"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])