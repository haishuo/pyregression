"""
Linear model solver.

Backend-agnostic linear regression algorithm.
"""

import numpy as np
from typing import Optional

from .qr import qr_decomposition_with_pivoting


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
    Fit linear model via QR decomposition.
    
    This is the core algorithm, backend-agnostic.
    
    Algorithm:
    ---------
    1. Add intercept to X
    2. Handle weights (if provided): scale X and y by sqrt(w)
    3. QR decomposition with column pivoting
    4. Solve R β = Q'y
    5. Compute residuals, fitted values
    6. Return results
    
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
    result : LinearModelResult
        Fitted model
    """
    from ..lm import LinearModelResult
    
    n = len(y)
    
    # Adjust for offset
    if offset is not None:
        y_adj = y - offset
    else:
        y_adj = y
    
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])
    p = X_with_intercept.shape[1]
    
    # Handle weights
    if weights is not None:
        # Separate zero and non-zero weights
        good = weights > 0
        n_good = np.sum(good)
        
        if n_good == 0:
            raise ValueError("All weights are zero")
        
        # Scale by sqrt(weights)
        w_sqrt = np.sqrt(weights[good])
        X_weighted = X_with_intercept[good, :] * w_sqrt[:, np.newaxis]
        y_weighted = y_adj[good] * w_sqrt
    else:
        X_weighted = X_with_intercept
        y_weighted = y_adj
        good = np.ones(n, dtype=bool)
        n_good = n
    
    # QR decomposition
    qr_result = qr_decomposition_with_pivoting(X_weighted, tol=tol, backend=backend)
    
    if not singular_ok and qr_result.rank < p:
        raise ValueError(f"Singular fit: rank {qr_result.rank} < {p} columns")
    
    # Solve R β = Q'y
    # TODO: Implement proper back-solve using backend
    # For now, placeholder
    from scipy.linalg import solve_triangular
    
    # Compute Q'y (need backend to apply Q)
    if backend is None:
        from .._backends import get_backend
        backend = get_backend('cpu')
    
    qty = backend.apply_qt_to_vector(qr_result.Q_implicit, y_weighted)
    
    # Back-solve
    rank = qr_result.rank
    coef_pivoted = np.zeros(p)
    if rank > 0:
        coef_pivoted[:rank] = solve_triangular(
            qr_result.R[:rank, :rank],
            qty[:rank],
            lower=False
        )
    
    # Unpivot coefficients
    coef = np.full(p, np.nan)
    pivot_idx = qr_result.pivot[:rank] - 1  # Convert to 0-indexed
    coef[pivot_idx] = coef_pivoted[:rank]
    
    # Compute fitted values and residuals
    fitted = X_with_intercept @ coef
    if offset is not None:
        fitted += offset
    
    residuals = y - fitted
    
    # Degrees of freedom
    df_residual = n - rank
    
    return LinearModelResult(
        coef=coef,
        residuals=residuals,
        fitted_values=fitted,
        rank=rank,
        df_residual=df_residual,
        qr_R=qr_result.R,
        qr_pivot=qr_result.pivot,
    )
