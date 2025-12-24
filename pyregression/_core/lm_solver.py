"""
Linear model solver.

Delegates to backend for actual computation.
"""

import numpy as np
from typing import Optional


def fit_linear_model(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    tol: Optional[float] = None,
    singular_ok: bool = True,
    backend = None,
):
    """
    Fit linear model via backend.
    
    This is just a thin wrapper - backends do all the work.
    
    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix (WITHOUT intercept)
    y : ndarray, shape (n,)
        Response vector
    weights : ndarray, optional
        Observation weights
    offset : ndarray, optional
        Offset term
    tol : float, optional
        Rank determination tolerance
    singular_ok : bool
        Allow singular fits
    backend : Backend, optional
        Computational backend
        
    Returns
    -------
    result : LinearModelResult (from backend)
        Fitted model
    """
    if backend is None:
        from .._backends import get_backend
        backend = get_backend('cpu')
    
    # Delegate everything to backend
    return backend.fit_linear_model(
        X, y,
        weights=weights,
        offset=offset,
        tol=tol,
        singular_ok=singular_ok
    )