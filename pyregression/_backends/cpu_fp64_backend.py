"""
CPU backend using NumPy + LAPACK.

This is the reference implementation validated against R.
"""

import numpy as np
from scipy.linalg.lapack import dgeqp3, dormqr
from typing import Optional

from .base import BackendBase, CPUBackend, QRResult


class CPUBackendFP64(CPUBackend):
    """
    CPU backend using NumPy + LAPACK.
    
    Reference implementation for R compatibility.
    Always uses FP64 precision.
    
    Algorithm:
    ---------
    QR decomposition via LAPACK dgeqp3 (Householder with column pivoting)
    """
    
    def __init__(self):
        self.name = "cpu_fp64"
        self.precision = "fp64"
    
    def qr_with_pivoting(self, X: np.ndarray, tol: Optional[float] = None) -> QRResult:
        """
        QR decomposition with column pivoting using LAPACK dgeqp3.
        
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
        Uses LAPACK dgeqp3 which replaces LINPACK dqrdc2.
        Numerically equivalent to R within 1e-12.
        """
        n, p = X.shape
        
        # Copy to Fortran order
        X_copy = np.asfortranarray(X.copy(), dtype=np.float64)
        
        # Default tolerance
        if tol is None:
            eps = np.finfo(np.float64).eps
            tol = max(n, p) * eps * np.linalg.norm(X_copy, 'fro')
        
        # Initialize pivot array
        jpvt = np.zeros(p, dtype=np.int32)
        
        # Call LAPACK dgeqp3
        qr, jpvt, tau, info = dgeqp3(X_copy, jpvt)
        
        if info < 0:
            raise ValueError(f"LAPACK dgeqp3 failed: illegal value at argument {-info}")
        
        # Determine rank
        R_diag = np.abs(np.diag(qr))
        if R_diag[0] == 0:
            rank = 0
        else:
            rank = np.sum(R_diag >= tol * R_diag[0])
        
        # Extract R (upper triangular part)
        R = np.triu(qr)
        
        # Store Q in implicit form (tau + qr lower part)
        Q_implicit = (qr, tau)
        
        # Convert pivot to 1-indexed (R convention)
        pivot = jpvt.astype(np.int64) + 1
        
        return QRResult(
            R=R,
            Q_implicit=Q_implicit,
            pivot=pivot,
            rank=rank,
            tol=tol,
        )
    
    def apply_qt_to_vector(self, Q_implicit: tuple, y: np.ndarray) -> np.ndarray:
        """
        Apply Q' to vector y.
        
        Parameters
        ----------
        Q_implicit : tuple
            (qr, tau) from LAPACK dgeqp3
        y : ndarray
            Vector to transform
            
        Returns
        -------
        qty : ndarray
            Q' @ y
        """
        qr, tau = Q_implicit
        
        # Copy y to Fortran order
        y_copy = np.asfortranarray(y.copy(), dtype=np.float64)
        
        # Apply Q' using LAPACK dormqr
        # side='L' (multiply from left), trans='T' (transpose Q)
        qty, info = dormqr(
            'L',  # side: multiply from Left
            'T',  # trans: Transpose (Q')
            qr,   # QR factorization
            tau,  # Householder scalars
            y_copy,  # vector to multiply
        )
        
        if info < 0:
            raise ValueError(f"LAPACK dormqr failed: illegal value at argument {-info}")
        
        return qty
    
    def get_device_info(self) -> dict:
        """Get backend information."""
        import scipy
        return {
            'backend': 'cpu',
            'precision': 'fp64',
            'library': f'NumPy {np.__version__}, SciPy {scipy.__version__}',
        }
