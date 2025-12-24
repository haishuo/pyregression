"""
GPU backend using PyTorch with FP32 precision.

Implements column-pivoted QR decomposition using Householder transformations.
This is a PyTorch translation of R's dqrdc2.f algorithm.
"""

import numpy as np
import warnings
from typing import Optional, Any

from .base import GPUBackendFP32, QRResult


class PyTorchBackendFP32(GPUBackendFP32):
    """
    PyTorch GPU backend with FP32 precision.
    
    Implements Householder QR with column pivoting for numerical stability.
    Algorithm based on R's dqrdc2.f (LINPACK with modifications).
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize PyTorch FP32 backend.
        
        Parameters
        ----------
        device : str, optional
            Device: 'cuda', 'mps', or None for auto-detect
        """
        self.name = "pytorch_fp32"
        self.precision = "fp32"
        
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch required for GPU backend. "
                "Install: pip install torch"
            )
        
        self.device = self._select_device(device)
        
    def _select_device(self, requested: Optional[str]) -> Any:
        """Select GPU device."""
        torch = self.torch
        
        if requested:
            return torch.device(requested)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            warnings.warn("No GPU available, using CPU")
            return torch.device('cpu')
    
    def qr_with_pivoting(self, X: np.ndarray, tol: Optional[float] = None) -> QRResult:
        """
        QR decomposition with column pivoting via Householder transformations.
        
        Algorithm: R's dqrdc2.f translated to PyTorch
        
        Parameters
        ----------
        X : ndarray, shape (n, p)
            Matrix to decompose
        tol : float, optional
            Tolerance for rank determination
            
        Returns
        -------
        QRResult
            QR decomposition with pivoting
        """
        torch = self.torch
        n, p = X.shape
        
        # Transfer to GPU as FP32
        X_work = torch.from_numpy(X.copy()).float().to(self.device)
        
        # Compute tolerance if not provided
        if tol is None:
            eps = torch.finfo(torch.float32).eps
            norm_X = torch.linalg.norm(X_work, 'fro')
            tol = max(n, p) * eps * norm_X
            tol = float(tol.item())
        
        # Initialize pivot indices (1-indexed like R)
        jpvt = torch.arange(1, p + 1, dtype=torch.int64, device=self.device)
        
        # Compute initial column norms
        qraux = torch.zeros(p, dtype=torch.float32, device=self.device)
        work_norm = torch.zeros(p, dtype=torch.float32, device=self.device)
        work_orig = torch.zeros(p, dtype=torch.float32, device=self.device)
        
        if n > 0:
            for j in range(p):
                col_norm = torch.linalg.norm(X_work[:, j])
                qraux[j] = col_norm
                work_norm[j] = col_norm
                work_orig[j] = col_norm
                if work_orig[j] == 0.0:
                    work_orig[j] = 1.0
        
        # Householder reduction
        lup = min(n, p)
        rank = p
        
        for l in range(lup):
            # Check if column l has negligible norm
            # If so, move it to the right (decrease rank)
            while l < rank and qraux[l] < work_orig[l] * tol:
                # Move column l to position rank-1
                self._move_column_right(X_work, jpvt, qraux, work_norm, work_orig, l, rank)
                rank -= 1
            
            if l >= rank or l == n:
                continue
            
            # Compute Householder transformation for column l
            nrmxl = torch.linalg.norm(X_work[l:n, l])
            
            if nrmxl == 0.0:
                continue
            
            # Sign of diagonal element
            if X_work[l, l] != 0.0:
                nrmxl = torch.sign(X_work[l, l]) * nrmxl
            
            # Scale column
            X_work[l:n, l] /= nrmxl
            X_work[l, l] += 1.0
            
            # Apply transformation to remaining columns
            if l + 1 < p:
                for j in range(l + 1, p):
                    # Compute: t = -<u, x_j> / u[0] where u = X_work[l:n, l]
                    t = -torch.dot(X_work[l:n, l], X_work[l:n, j]) / X_work[l, l]
                    
                    # Apply: x_j += t * u
                    X_work[l:n, j] += t * X_work[l:n, l]
                    
                    # Update column norm
                    if qraux[j] != 0.0:
                        tt = 1.0 - (torch.abs(X_work[l, j]) / qraux[j]) ** 2
                        tt = torch.clamp(tt, min=0.0)
                        
                        # R's stability check: recompute if large reduction
                        if torch.abs(tt) < 1e-6:
                            # Recompute norm from scratch
                            if l + 1 < n:
                                qraux[j] = torch.linalg.norm(X_work[l+1:n, j])
                                work_norm[j] = qraux[j]
                        else:
                            # Incremental update
                            qraux[j] *= torch.sqrt(tt)
            
            # Save the transformation
            qraux[l] = X_work[l, l]
            X_work[l, l] = -nrmxl
        
        rank = min(rank, n)
        
        # Extract R (upper triangular part)
        R = torch.triu(X_work[:p, :p])
        
        # Create full Q matrix from Householder reflectors
        Q = self._form_q_from_householder(X_work, qraux, n, p, min(n, p))
        
        # Transfer back to numpy
        return QRResult(
            R=R.cpu().numpy(),
            Q_implicit=Q.cpu().numpy(),
            pivot=jpvt.cpu().numpy(),
            rank=rank,
            tol=tol
        )
    
    def _move_column_right(self, X, jpvt, qraux, work_norm, work_orig, l, p_current):
        """
        Move column l to the right edge (position p_current-1).
        
        This implements R's strategy for handling rank-deficient matrices.
        """
        # Shift columns [l+1:p_current] left by one
        X[:, l:p_current-1] = X[:, l+1:p_current].clone()
        
        # Move column l data to the end
        temp_col = X[:, l].clone()
        X[:, p_current-1] = temp_col
        
        # Update pivot indices
        temp_pivot = jpvt[l].clone()
        jpvt[l:p_current-1] = jpvt[l+1:p_current].clone()
        jpvt[p_current-1] = temp_pivot
        
        # Update norms
        temp_qr = qraux[l].clone()
        temp_wn = work_norm[l].clone()
        temp_wo = work_orig[l].clone()
        
        qraux[l:p_current-1] = qraux[l+1:p_current].clone()
        work_norm[l:p_current-1] = work_norm[l+1:p_current].clone()
        work_orig[l:p_current-1] = work_orig[l+1:p_current].clone()
        
        qraux[p_current-1] = temp_qr
        work_norm[p_current-1] = temp_wn
        work_orig[p_current-1] = temp_wo
    
    def _form_q_from_householder(self, X, qraux, n, p, num_reflectors):
        """
        Form full Q matrix from Householder reflectors stored in X.
        
        Q = H_1 * H_2 * ... * H_k where H_i = I - beta * u_i * u_i'
        """
        torch = self.torch
        
        # Start with identity
        Q = torch.eye(n, dtype=torch.float32, device=self.device)
        
        # Apply Householder reflectors in reverse order
        for k in range(min(num_reflectors, n) - 1, -1, -1):
            if k >= n or qraux[k] == 0.0:
                continue
            
            # Householder vector: u = [X[k, k], X[k+1:n, k]]
            # But X[k, k] was saved in qraux[k]
            u = torch.zeros(n - k, dtype=torch.float32, device=self.device)
            u[0] = qraux[k]
            if k + 1 < n:
                u[1:] = X[k+1:n, k]
            
            # Apply H = I - beta * u * u' to Q[k:n, :]
            # beta = 2 / ||u||^2, but u[0] includes the +1, so beta = 2/u[0]
            beta = 2.0 / u[0] if u[0] != 0.0 else 0.0
            
            # Q[k:n, :] = Q[k:n, :] - beta * u * (u' * Q[k:n, :])
            temp = beta * torch.outer(u, torch.matmul(u, Q[k:n, :]))
            Q[k:n, :] -= temp
        
        return Q
    
    def apply_qt_to_vector(self, Q_implicit: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply Q' to vector y.
        
        Parameters
        ----------
        Q_implicit : ndarray
            Full Q matrix (already transferred back from GPU)
        y : ndarray
            Vector to transform
            
        Returns
        -------
        ndarray
            Q' @ y
        """
        # Simple matrix multiplication (Q already on CPU)
        return Q_implicit.T @ y
    
    def get_device_info(self) -> dict:
        """Get backend information."""
        return {
            'backend': 'gpu',
            'precision': 'fp32',
            'device': str(self.device),
            'library': f'PyTorch {self.torch.__version__}',
        }