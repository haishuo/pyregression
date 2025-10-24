"""
QR decomposition with column pivoting.

Backend-agnostic interface to QR factorization.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class QRDecomposition:
    """Result of QR decomposition with pivoting."""
    R: np.ndarray            # Upper triangular matrix R
    Q_implicit: np.ndarray   # Q in implicit form (for back-transformation)
    pivot: np.ndarray        # Pivot indices (1-indexed, R convention)
    rank: int                # Determined rank
    tol: float               # Tolerance used


def qr_decomposition_with_pivoting(
    X: np.ndarray,
    tol: Optional[float] = None,
    backend = None,
) -> QRDecomposition:
    """
    QR decomposition with column pivoting.
    
    Delegates to backend-specific implementation.
    
    Parameters
    ----------
    X : ndarray, shape (n, p)
        Matrix to decompose
    tol : float, optional
        Tolerance for rank determination
    backend : Backend, optional
        Computational backend
        
    Returns
    -------
    result : QRDecomposition
        QR decomposition with pivoting
    """
    if backend is None:
        from .._backends import get_backend
        backend = get_backend('cpu')
    
    return backend.qr_with_pivoting(X, tol=tol)
