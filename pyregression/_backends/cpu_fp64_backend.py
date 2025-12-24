"""
CPU backend using NumPy + SciPy.

This is the reference implementation validated against R.
"""

import numpy as np
from scipy.linalg import qr, solve_triangular
from typing import Optional

from .base import CPUBackend, LinearModelResult


class CPUBackendFP64(CPUBackend):
    """
    CPU backend using NumPy + SciPy.
    
    Reference implementation for R compatibility.
    Always uses FP64 precision.
    """
    
    def __init__(self):
        self.name = "cpu_fp64"
        self.precision = "fp64"
    
    def fit_linear_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        tol: Optional[float] = None,
        singular_ok: bool = True
    ) -> LinearModelResult:
        """
        Fit linear model using NumPy/LAPACK.
        
        Complete implementation - all computation stays in NumPy.
        """
        n = len(y)
        
        # Adjust for offset
        y_work = y - offset if offset is not None else y.copy()
        
        # Add intercept
        X_work = np.column_stack([np.ones(n), X])
        p = X_work.shape[1]
        
        # Handle weights
        if weights is not None:
            good = weights > 0
            if not np.any(good):
                raise ValueError("All weights are zero")
            
            w_sqrt = np.sqrt(weights[good])
            X_work = X_work[good, :] * w_sqrt[:, np.newaxis]
            y_work = y_work[good] * w_sqrt
            n_good = np.sum(good)
        else:
            good = np.ones(n, dtype=bool)
            n_good = n
        
        # Ensure float64
        X_work = np.asarray(X_work, dtype=np.float64)
        y_work = np.asarray(y_work, dtype=np.float64)
        
        # Compute tolerance
        if tol is None:
            eps = np.finfo(np.float64).eps
            tol = max(n_good, p) * eps * np.linalg.norm(X_work, 'fro')
        
        # QR decomposition with column pivoting
        Q, R, P = qr(X_work, mode='full', pivoting=True)
        
        # Determine rank
        R_diag = np.abs(np.diag(R))
        if R_diag[0] == 0:
            rank = 0
        else:
            rank = np.sum(R_diag >= tol * R_diag[0])
        
        if not singular_ok and rank < p:
            raise ValueError(f"Singular fit: rank {rank} < {p} columns")
        
        # Solve R β = Q'y
        qty = Q.T @ y_work
        
        # Initialize coefficients (with NaN for aliased)
        coef = np.full(p, np.nan, dtype=np.float64)
        
        if rank > 0:
            # Back-solve for non-aliased coefficients
            coef_active = solve_triangular(
                R[:rank, :rank],
                qty[:rank],
                lower=False
            )
            coef[P[:rank]] = coef_active
        
        # Compute fitted values and residuals on original scale
        X_full = np.column_stack([np.ones(n), X])
        
        # Compute fitted values (handling NaN coefficients for aliased terms)
        valid_coef = ~np.isnan(coef)
        if np.any(valid_coef):
            fitted = X_full[:, valid_coef] @ coef[valid_coef]
        else:
            fitted = np.zeros(n, dtype=np.float64)
        
        residuals = y - fitted
        
        # Adjust fitted values for offset
        if offset is not None:
            fitted += offset
        
        return LinearModelResult(
            coef=coef,
            residuals=residuals,
            fitted_values=fitted,
            rank=rank,
            df_residual=n_good - rank,
            qr_R=R[:p, :p],
            qr_pivot=P.astype(np.int64) + 1,  # 1-indexed like R
            qr_tol=tol
        )
    
    def get_device_info(self) -> dict:
        """Get backend information."""
        import scipy
        return {
            'backend': 'cpu',
            'precision': 'fp64',
            'library': f'NumPy {np.__version__}, SciPy {scipy.__version__}',
        }