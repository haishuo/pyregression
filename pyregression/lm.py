"""
Linear model API.

Main user-facing interface for linear regression.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from ._backends import get_backend
from ._backends.base import LinearModelResult as BackendResult
from ._core.lm_solver import fit_linear_model


@dataclass
class LinearModelResult(BackendResult):
    """
    Results from linear regression.
    
    Extends backend result with computed statistics.
    """
    
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
        from scipy.linalg import solve_triangular
        
        n = len(self.residuals)
        rss = np.sum(self.residuals ** 2)
        sigma_sq = rss / self.df_residual if self.df_residual > 0 else 0.0
        
        # Compute (R'R)^-1
        R_inv = solve_triangular(
            self.qr_R[:self.rank, :self.rank],
            np.eye(self.rank),
            lower=False
        )
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
        y = self.fitted_values + self.residuals
        rss = np.sum(self.residuals ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        
        self._r_squared = 1 - rss / tss if tss > 0 else 0.0
        
        n = len(self.residuals)
        if self.df_residual > 0:
            self._adj_r_squared = 1 - (1 - self._r_squared) * (n - 1) / self.df_residual
        else:
            self._adj_r_squared = np.nan


class LinearModel:
    """Linear regression via QR decomposition."""
    
    def __init__(self, backend: str = 'auto', use_fp64: Optional[bool] = None):
        """Initialize linear model."""
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
        """Fit linear model."""
        backend_result = fit_linear_model(
            X, y,
            weights=weights,
            offset=offset,
            tol=tol,
            singular_ok=singular_ok,
            backend=self.backend
        )
        
        # Convert to user-facing result with lazy statistics
        return LinearModelResult(**backend_result.__dict__)