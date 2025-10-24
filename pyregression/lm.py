"""
Linear model API.

Main user-facing interface for linear regression.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from ._backends import get_backend
from ._core.lm_solver import fit_linear_model


@dataclass
class LinearModelResult:
    """Results from linear regression."""
    coef: np.ndarray          # Coefficients
    residuals: np.ndarray     # Residuals
    fitted_values: np.ndarray # Fitted values
    rank: int                 # Rank of design matrix
    df_residual: int          # Residual degrees of freedom
    
    # QR components
    qr_R: np.ndarray          # R from QR decomposition
    qr_pivot: np.ndarray      # Pivot indices
    
    # Statistics (computed lazily)
    _se: Optional[np.ndarray] = None
    _vcov: Optional[np.ndarray] = None
    _r_squared: Optional[float] = None
    _adj_r_squared: Optional[float] = None
    
    @property
    def se(self) -> np.ndarray:
        """Standard errors of coefficients."""
        if self._se is None:
            self._compute_statistics()
        return self._se
    
    @property
    def vcov(self) -> np.ndarray:
        """Variance-covariance matrix."""
        if self._vcov is None:
            self._compute_statistics()
        return self._vcov
    
    @property
    def r_squared(self) -> float:
        """R-squared."""
        if self._r_squared is None:
            self._compute_r_squared()
        return self._r_squared
    
    @property
    def adj_r_squared(self) -> float:
        """Adjusted R-squared."""
        if self._adj_r_squared is None:
            self._compute_r_squared()
        return self._adj_r_squared
    
    def _compute_statistics(self):
        """Compute standard errors and covariance matrix."""
        # TODO: Implement after QR solver complete
        n = len(self.residuals)
        rss = np.sum(self.residuals ** 2)
        sigma_sq = rss / self.df_residual if self.df_residual > 0 else 0.0
        
        # Compute (R'R)^-1
        from scipy.linalg import solve_triangular
        R_inv = solve_triangular(self.qr_R[:self.rank, :self.rank], 
                                  np.eye(self.rank), lower=False)
        vcov_full = R_inv @ R_inv.T * sigma_sq
        
        # Expand to full size with NAs for aliased
        p = len(self.coef)
        vcov = np.full((p, p), np.nan)
        pivot_idx = self.qr_pivot[:self.rank] - 1  # Convert to 0-indexed
        vcov[np.ix_(pivot_idx, pivot_idx)] = vcov_full
        
        self._vcov = vcov
        self._se = np.sqrt(np.diag(vcov))
    
    def _compute_r_squared(self):
        """Compute R-squared statistics."""
        # TODO: Handle intercept correctly
        y = self.fitted_values + self.residuals
        rss = np.sum(self.residuals ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        
        self._r_squared = 1 - rss / tss if tss > 0 else 0.0
        
        n = len(self.residuals)
        p = self.rank
        if self.df_residual > 0:
            self._adj_r_squared = 1 - (1 - self._r_squared) * (n - 1) / self.df_residual
        else:
            self._adj_r_squared = np.nan


class LinearModel:
    """
    Linear regression via QR decomposition.
    
    Provides R-compatible linear regression with optional GPU acceleration.
    
    Examples
    --------
    >>> import numpy as np
    >>> from pyregression import LinearModel
    >>> 
    >>> X = np.random.randn(100, 5)
    >>> y = X @ np.array([1, -0.5, 0.3, 0, 0.8]) + np.random.randn(100)
    >>> 
    >>> model = LinearModel()
    >>> result = model.fit(X, y)
    >>> print(result.coef)
    >>> print(result.se)
    >>> print(result.r_squared)
    """
    
    def __init__(self, backend: str = 'auto', use_fp64: Optional[bool] = None):
        """
        Initialize linear model.
        
        Parameters
        ----------
        backend : str, default='auto'
            Backend to use: 'auto', 'cpu', 'gpu'
        use_fp64 : bool, optional
            Use FP64 precision. If None, auto-select based on hardware.
        """
        self.backend = get_backend(backend, use_fp64=use_fp64)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        tol: Optional[float] = None,
        singular_ok: bool = True,
    ) -> LinearModelResult:
        """
        Fit linear model.
        
        Parameters
        ----------
        X : ndarray, shape (n, p)
            Design matrix (without intercept - added automatically)
        y : ndarray, shape (n,)
            Response vector
        weights : ndarray, shape (n,), optional
            Observation weights for WLS
        offset : ndarray, shape (n,), optional
            Offset term
        tol : float, optional
            Tolerance for rank determination.
            Default: max(n,p) * eps * ||X||_F
        singular_ok : bool, default=True
            If False, raise error on singular fit
            
        Returns
        -------
        result : LinearModelResult
            Fitted model results
            
        Notes
        -----
        Uses QR decomposition with column pivoting via LAPACK.
        Numerically equivalent to R's lm() within 1e-12.
        """
        return fit_linear_model(
            X, y,
            weights=weights,
            offset=offset,
            tol=tol,
            singular_ok=singular_ok,
            backend=self.backend
        )
