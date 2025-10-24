"""
Generalized linear model API.

Main user-facing interface for GLMs.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from ._backends import get_backend
from ._core.families import Family


@dataclass
class GLMResult:
    """Results from GLM fitting."""
    coef: np.ndarray          # Coefficients
    residuals: np.ndarray     # Residuals (response scale)
    fitted_values: np.ndarray # Fitted values (μ)
    linear_predictors: np.ndarray  # Linear predictors (η)
    
    rank: int                 # Rank
    df_residual: int          # Residual df
    
    deviance: float           # Deviance
    null_deviance: float      # Null deviance
    aic: float                # AIC
    
    converged: bool           # Converged?
    boundary: bool            # Hit boundary?
    iterations: int           # IRLS iterations
    
    _se: Optional[np.ndarray] = None


class GLM:
    """
    Generalized linear model via IRLS.
    
    Provides R-compatible GLMs with optional GPU acceleration.
    """
    
    def __init__(self, family: Family, backend: str = 'auto', 
                 use_fp64: Optional[bool] = None):
        """
        Initialize GLM.
        
        Parameters
        ----------
        family : Family
            GLM family (Gaussian, Binomial, Poisson, etc.)
        backend : str, default='auto'
            Backend: 'auto', 'cpu', 'gpu'
        use_fp64 : bool, optional
            Use FP64 precision
        """
        self.family = family
        self.backend = get_backend(backend, use_fp64=use_fp64)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> GLMResult:
        """
        Fit generalized linear model.
        
        Parameters
        ----------
        X : ndarray, shape (n, p)
            Design matrix
        y : ndarray, shape (n,)
            Response vector
        **kwargs
            Additional arguments
            
        Returns
        -------
        result : GLMResult
            Fitted model results
        """
        # TODO: Implement IRLS
        raise NotImplementedError("GLM fitting not yet implemented")
