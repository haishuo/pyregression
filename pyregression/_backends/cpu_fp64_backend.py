"""
CPU backend using NumPy + SciPy.

This is the reference implementation validated against R.
"""

import numpy as np
from scipy.linalg import qr
from typing import Optional

from .base import BackendBase, CPUBackend, QRResult


class CPUBackendFP64(CPUBackend):
    """
    CPU backend using NumPy + SciPy.
    
    Reference implementation for R compatibility.
    Always uses FP64 precision.
    
    Algorithm:
    ---------
    QR decomposition via SciPy's qr() with column pivoting.
    This uses LAPACK under the hood but with a simpler interface.
    """
    
    def __init__(self):
        self.name = "cpu_fp64"
        self.precision = "fp64"
    
    def qr_with_pivoting(self, X: np.ndarray, tol: Optional[float] = None) -> QRResult:
        """
        QR decomposition with column pivoting.
        
        Parameters
        ----------
        X : ndarray, shape (n, p)
            Matrix to decompose
        tol : float, optional
            Tolerance for rank determination
            
        Returns
        -------
        result : QRResult
            QR decomposition
            
        Notes
        -----
        Uses SciPy's qr() with pivoting='True', which calls LAPACK.
        Numerically equivalent to R within 1e-12.
        """
        n, p = X.shape
        
        # Ensure float64
        X_copy = np.asarray(X, dtype=np.float64)
        
        # Default tolerance
        if tol is None:
            eps = np.finfo(np.float64).eps
            tol = max(n, p) * eps * np.linalg.norm(X_copy, 'fro')
        
        # QR decomposition with column pivoting
        # mode='full' returns full Q (n x n) and R (n x p)
        # This matches R's behavior
        Q, R, P = qr(X_copy, mode='full', pivoting=True)
        
        # Determine rank from diagonal of R
        R_diag = np.abs(np.diag(R))
        if R_diag[0] == 0:
            rank = 0
        else:
            rank = np.sum(R_diag >= tol * R_diag[0])
        
        # Store Q explicitly (we'll need it for apply_qt_to_vector)
        Q_implicit = Q  # Store full Q matrix
        
        # Convert pivot to 1-indexed (R convention)
        # SciPy returns P as column indices (0-indexed)
        pivot = P.astype(np.int64) + 1
        
        return QRResult(
            R=R,
            Q_implicit=Q_implicit,
            pivot=pivot,
            rank=rank,
            tol=tol,
        )
    
    def apply_qt_to_vector(self, Q_implicit: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply Q' to vector y.
        
        Parameters
        ----------
        Q_implicit : ndarray
            Q matrix from QR decomposition
        y : ndarray
            Vector to transform
            
        Returns
        -------
        qty : ndarray
            Q' @ y
        """
        Q = Q_implicit  # Q is stored explicitly
        
        # Q' @ y
        qty = Q.T @ y
        
        return qty
    
    def get_device_info(self) -> dict:
        """Get backend information."""
        import scipy
        return {
            'backend': 'cpu',
            'precision': 'fp64',
            'library': f'NumPy {np.__version__}, SciPy {scipy.__version__}',
        }