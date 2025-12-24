"""
Linear regression with R-style interface and output.

This is the user-facing API that statisticians actually use.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
from dataclasses import dataclass
from scipy import stats

from ._backends import get_backend


class LinearModel:
    """
    Fit linear regression model (like R's lm()).
    
    This provides a complete statistical interface, not just QR outputs.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from pyregression import lm
    >>> 
    >>> # Load data
    >>> data = pd.read_csv('trial_data.csv')
    >>> 
    >>> # Fit model (R-style formula interface coming soon!)
    >>> model = lm(y='sbp_change', 
    ...           X=['treatment', 'age', 'baseline_sbp'], 
    ...           data=data)
    >>> 
    >>> # See results
    >>> model.summary()  # Prints nice table like R
    >>> 
    >>> # Extract what you need
    >>> model.coef         # Named coefficients
    >>> model.pvalues      # P-values for each coefficient
    >>> model.conf_int()   # Confidence intervals
    >>> model.predict(new_data)  # Make predictions
    """
    
    def __init__(
        self,
        y: Union[str, np.ndarray],
        X: Union[List[str], np.ndarray],
        data: Optional[pd.DataFrame] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
        backend: str = 'auto',
        use_fp64: Optional[bool] = None
    ):
        """
        Fit linear regression model.
        
        Parameters
        ----------
        y : str or array
            Response variable (outcome)
            - If string: column name in data
            - If array: numeric values
        X : list of str or array
            Predictor variables (covariates)
            - If list of strings: column names in data
            - If array: numeric matrix (n × p)
        data : DataFrame, optional
            Dataset containing y and X variables
        weights : str or array, optional
            Observation weights
        backend : str
            Computational backend: 'auto', 'cpu', 'gpu'
        use_fp64 : bool, optional
            Force double precision (for regulatory compliance)
        
        Examples
        --------
        >>> # Using pandas DataFrame (recommended)
        >>> model = lm(y='mpg', X=['wt', 'hp', 'cyl'], data=mtcars)
        >>> model.summary()
        
        >>> # Using arrays (if you must)
        >>> model = lm(y=y_array, X=X_matrix)
        """
        # Parse inputs
        if isinstance(y, str):
            if data is None:
                raise ValueError("Must provide data when y is a string")
            self.y_values = data[y].values
            self.y_name = y
        else:
            self.y_values = np.asarray(y)
            self.y_name = 'y'
        
        if isinstance(X, list) and all(isinstance(x, str) for x in X):
            if data is None:
                raise ValueError("Must provide data when X is list of strings")
            self.X_values = data[X].values
            self.X_names = X
        else:
            self.X_values = np.asarray(X)
            self.X_names = [f'x{i}' for i in range(self.X_values.shape[1])]
        
        if weights is not None:
            if isinstance(weights, str):
                if data is None:
                    raise ValueError("Must provide data when weights is a string")
                self.weights_values = data[weights].values
            else:
                self.weights_values = np.asarray(weights)
        else:
            self.weights_values = None
        
        # Store metadata
        self.n_obs = len(self.y_values)
        self.n_coef = self.X_values.shape[1] + 1  # +1 for intercept
        self.var_names = ['Intercept'] + self.X_names
        
        # Fit model using backend
        self.backend = get_backend(backend, use_fp64=use_fp64)
        self._backend_result = self.backend.fit_linear_model(
            self.X_values, 
            self.y_values, 
            weights=self.weights_values
        )
        
        # Compute statistical inference
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute standard errors, t-stats, p-values, etc."""
        result = self._backend_result
        
        # Extract from backend
        self.coefficients = result.coef
        self.residuals = result.residuals
        self.fitted_values = result.fitted_values
        self.rank = result.rank
        self.df_residual = result.df_residual
        
        # Residual standard error
        rss = np.sum(self.residuals**2)
        self.sigma = np.sqrt(rss / self.df_residual)
        
        # Variance-covariance matrix of coefficients
        # Var(β) = σ² (X'X)⁻¹
        R = result.qr_R[:self.rank, :self.rank]
        R_inv = np.linalg.inv(R)
        XtX_inv = R_inv @ R_inv.T
        
        # Reorder based on pivot
        pivot = result.qr_pivot[:self.rank] - 1
        var_beta = np.full((self.n_coef, self.n_coef), np.nan)
        var_beta_active = XtX_inv * (self.sigma ** 2)
        
        # Place active coefficients in correct positions
        for i, pi in enumerate(pivot):
            for j, pj in enumerate(pivot):
                var_beta[pi, pj] = var_beta_active[i, j]
        
        self.vcov = var_beta
        
        # Standard errors
        self.std_errors = np.sqrt(np.diag(self.vcov))
        
        # t-statistics
        self.t_values = self.coefficients / self.std_errors
        
        # p-values (two-tailed)
        self.pvalues = 2 * (1 - stats.t.cdf(np.abs(self.t_values), self.df_residual))
        
        # R-squared
        tss = np.sum((self.y_values - np.mean(self.y_values))**2)
        self.r_squared = 1 - (rss / tss) if tss > 0 else 0.0
        
        # Adjusted R-squared
        n = self.n_obs
        p = self.rank - 1  # Exclude intercept
        if self.df_residual > 0:
            self.adj_r_squared = 1 - (1 - self.r_squared) * (n - 1) / self.df_residual
        else:
            self.adj_r_squared = np.nan
        
        # F-statistic
        if p > 0 and self.df_residual > 0:
            self.f_statistic = ((tss - rss) / p) / (rss / self.df_residual)
            self.f_pvalue = 1 - stats.f.cdf(self.f_statistic, p, self.df_residual)
        else:
            self.f_statistic = np.nan
            self.f_pvalue = np.nan
    
    @property
    def coef(self):
        """Named coefficients (pandas Series)."""
        return pd.Series(self.coefficients, index=self.var_names)
    
    def conf_int(self, alpha: float = 0.05):
        """
        Confidence intervals for coefficients.
        
        Parameters
        ----------
        alpha : float
            Significance level (default: 0.05 for 95% CI)
        
        Returns
        -------
        DataFrame
            Confidence intervals with columns 'lower' and 'upper'
        """
        t_crit = stats.t.ppf(1 - alpha/2, self.df_residual)
        lower = self.coefficients - t_crit * self.std_errors
        upper = self.coefficients + t_crit * self.std_errors
        
        return pd.DataFrame({
            'lower': lower,
            'upper': upper
        }, index=self.var_names)
    
    def summary(self):
        """
        Print summary of regression results (like R's summary.lm).
        
        This is what statisticians actually want to see.
        """
        print()
        print("="*80)
        print("LINEAR REGRESSION RESULTS")
        print("="*80)
        print()
        
        # Model info
        print(f"Dependent variable: {self.y_name}")
        print(f"Number of observations: {self.n_obs}")
        print(f"Degrees of freedom: {self.df_residual} (residual), {self.rank - 1} (model)")
        print()
        
        # Residuals
        print("Residuals:")
        residual_summary = pd.Series(self.residuals).describe()
        print(f"  Min:    {residual_summary['min']:>10.4f}")
        print(f"  1Q:     {residual_summary['25%']:>10.4f}")
        print(f"  Median: {residual_summary['50%']:>10.4f}")
        print(f"  3Q:     {residual_summary['75%']:>10.4f}")
        print(f"  Max:    {residual_summary['max']:>10.4f}")
        print()
        
        # Coefficients table
        print("Coefficients:")
        print("-"*80)
        print(f"{'Variable':<20} {'Estimate':>12} {'Std. Error':>12} {'t value':>10} {'Pr(>|t|)':>12}")
        print("-"*80)
        
        for i, name in enumerate(self.var_names):
            # Significance stars
            p = self.pvalues[i]
            if np.isnan(p):
                sig = ' (aliased)'
                p_str = 'NA'
            else:
                if p < 0.001:
                    sig = ' ***'
                elif p < 0.01:
                    sig = ' **'
                elif p < 0.05:
                    sig = ' *'
                elif p < 0.1:
                    sig = ' .'
                else:
                    sig = ''
                
                p_str = f"{p:.4f}" if p >= 0.0001 else "<.0001"
            
            print(f"{name:<20} {self.coefficients[i]:>12.4f} {self.std_errors[i]:>12.4f} "
                  f"{self.t_values[i]:>10.3f} {p_str:>12}{sig}")
        
        print("-"*80)
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print()
        
        # Model fit statistics
        print(f"Residual standard error: {self.sigma:.4f} on {self.df_residual} degrees of freedom")
        print(f"Multiple R-squared:      {self.r_squared:.4f}")
        print(f"Adjusted R-squared:      {self.adj_r_squared:.4f}")
        
        if not np.isnan(self.f_statistic):
            f_pval_str = f"{self.f_pvalue:.4e}" if self.f_pvalue >= 0.0001 else "< 2.2e-16"
            print(f"F-statistic:             {self.f_statistic:.2f} on {self.rank-1} and {self.df_residual} DF, p-value: {f_pval_str}")
        
        print()
        print(f"Backend: {self.backend.name}")
        print("="*80)
        print()
    
    def predict(self, newdata: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict response for new data.
        
        Parameters
        ----------
        newdata : DataFrame or array
            New predictor values
            - If DataFrame: must have columns matching self.X_names
            - If array: must have same number of columns as X
        
        Returns
        -------
        array
            Predicted values
        """
        if isinstance(newdata, pd.DataFrame):
            X_new = newdata[self.X_names].values
        else:
            X_new = np.asarray(newdata)
        
        # Add intercept
        X_new_full = np.column_stack([np.ones(len(X_new)), X_new])
        
        # Predict (handling NaN coefficients for aliased terms)
        valid = ~np.isnan(self.coefficients)
        predictions = X_new_full[:, valid] @ self.coefficients[valid]
        
        return predictions
    
    def __repr__(self):
        return f"LinearModel(n={self.n_obs}, p={self.rank-1}, R²={self.r_squared:.3f})"


def lm(y, X, data=None, **kwargs):
    """
    Fit linear regression model (convenience function).
    
    This is the main function statisticians should use.
    It's designed to feel like R's lm() but with Python/pandas.
    
    Parameters
    ----------
    y : str or array
        Response variable
    X : list of str or array
        Predictor variables
    data : DataFrame, optional
        Dataset
    **kwargs
        Additional arguments passed to LinearModel
    
    Returns
    -------
    LinearModel
        Fitted model object
    
    Examples
    --------
    >>> # Basic usage (pandas)
    >>> model = lm(y='mpg', X=['wt', 'hp'], data=mtcars)
    >>> model.summary()
    >>> 
    >>> # Get coefficients
    >>> model.coef
    >>> 
    >>> # Get p-values
    >>> model.pvalues
    >>> 
    >>> # Confidence intervals
    >>> model.conf_int()
    >>> 
    >>> # Make predictions
    >>> new_cars = pd.DataFrame({'wt': [3.0, 3.5], 'hp': [110, 150]})
    >>> model.predict(new_cars)
    """
    return LinearModel(y=y, X=X, data=data, **kwargs)