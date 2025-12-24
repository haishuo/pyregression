"""
GPU backend using PyTorch with FP64 precision.

For data center GPUs: A100, H100, V100.
"""

import numpy as np
import warnings
from typing import Optional

from .gpu_fp32_backend import PyTorchBackendFP32
from .base import GPUBackendFP64, QRResult


class PyTorchBackendFP64(GPUBackendFP64):
    """
    PyTorch GPU backend with FP64 precision.
    
    Inherits from FP32 backend but uses float64 precision.
    Only recommended for data center GPUs with full FP64 support.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize PyTorch FP64 backend.
        
        Parameters
        ----------
        device : str, optional
            Device: 'cuda' or None for auto-detect
            
        Raises
        ------
        RuntimeError
            If device is 'mps' (Metal doesn't support FP64)
        """
        # Initialize like FP32 but we'll override precision
        self.name = "pytorch_fp64"
        self.precision = "fp64"
        
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch required for GPU backend. "
                "Install: pip install torch"
            )
        
        # Device selection
        if device == 'mps':
            raise RuntimeError(
                "FP64 not supported on Apple Metal. "
                "Use FP32 backend or CPU."
            )
        
        if device is None:
            # Auto-detect (only CUDA, no Metal for FP64)
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                warnings.warn("No CUDA GPU available, using CPU")
                device = 'cpu'
        
        self.device = torch.device(device)
        
        # Warn if using FP64 on gimped hardware
        if device == 'cuda':
            from .precision_detector import detect_gpu_capabilities
            caps = detect_gpu_capabilities()
            if caps.fp64_support.value == 'gimped_fp64':
                warnings.warn(
                    f"Using FP64 on {caps.gpu_name} with gimped FP64 support. "
                    f"This will be ~{int(1/caps.fp64_throughput_ratio)}x slower than FP32. "
                    f"Consider using FP32 for better performance.",
                    UserWarning
                )
    
    def qr_with_pivoting(self, X: np.ndarray, tol: Optional[float] = None) -> QRResult:
        """
        QR decomposition with FP64 precision.
        
        Same algorithm as FP32 but using float64.
        """
        torch = self.torch
        n, p = X.shape
        
        # Transfer to GPU as FP64 (only difference from FP32)
        X_work = torch.from_numpy(X.copy()).double().to(self.device)
        
        # Compute tolerance
        if tol is None:
            eps = torch.finfo(torch.float64).eps
            norm_X = torch.linalg.norm(X_work, 'fro')
            tol = max(n, p) * eps * norm_X
            tol = float(tol.item())
        
        # Rest is identical to FP32 implementation
        # (We could refactor to share code, but for now just duplicate)
        
        # Initialize pivot indices
        jpvt = torch.arange(1, p + 1, dtype=torch.int64, device=self.device)
        
        # Compute initial column norms
        qraux = torch.zeros(p, dtype=torch.float64, device=self.device)
        work_norm = torch.zeros(p, dtype=torch.float64, device=self.device)
        work_orig = torch.zeros(p, dtype=torch.float64, device=self.device)
        
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
            while l < rank and qraux[l] < work_orig[l] * tol:
                self._move_column_right(X_work, jpvt, qraux, work_norm, work_orig, l, rank)
                rank -= 1
            
            if l >= rank or l == n:
                continue
            
            nrmxl = torch.linalg.norm(X_work[l:n, l])
            
            if nrmxl == 0.0:
                continue
            
            if X_work[l, l] != 0.0:
                nrmxl = torch.sign(X_work[l, l]) * nrmxl
            
            X_work[l:n, l] /= nrmxl
            X_work[l, l] += 1.0
            
            if l + 1 < p:
                for j in range(l + 1, p):
                    t = -torch.dot(X_work[l:n, l], X_work[l:n, j]) / X_work[l, l]
                    X_work[l:n, j] += t * X_work[l:n, l]
                    
                    if qraux[j] != 0.0:
                        tt = 1.0 - (torch.abs(X_work[l, j]) / qraux[j]) ** 2
                        tt = torch.clamp(tt, min=0.0)
                        
                        if torch.abs(tt) < 1e-6:
                            if l + 1 < n:
                                qraux[j] = torch.linalg.norm(X_work[l+1:n, j])
                                work_norm[j] = qraux[j]
                        else:
                            qraux[j] *= torch.sqrt(tt)
            
            qraux[l] = X_work[l, l]
            X_work[l, l] = -nrmxl
        
        rank = min(rank, n)
        
        R = torch.triu(X_work[:p, :p])
        Q = self._form_q_from_householder(X_work, qraux, n, p, min(n, p))
        
        return QRResult(
            R=R.cpu().numpy(),
            Q_implicit=Q.cpu().numpy(),
            pivot=jpvt.cpu().numpy(),
            rank=rank,
            tol=tol
        )
    
    def _move_column_right(self, X, jpvt, qraux, work_norm, work_orig, l, p_current):
        """Move column to right (same as FP32)."""
        X[:, l:p_current-1] = X[:, l+1:p_current].clone()
        temp_col = X[:, l].clone()
        X[:, p_current-1] = temp_col
        
        temp_pivot = jpvt[l].clone()
        jpvt[l:p_current-1] = jpvt[l+1:p_current].clone()
        jpvt[p_current-1] = temp_pivot
        
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
        """Form Q matrix (same as FP32)."""
        torch = self.torch
        
        Q = torch.eye(n, dtype=torch.float64, device=self.device)
        
        for k in range(min(num_reflectors, n) - 1, -1, -1):
            if k >= n or qraux[k] == 0.0:
                continue
            
            u = torch.zeros(n - k, dtype=torch.float64, device=self.device)
            u[0] = qraux[k]
            if k + 1 < n:
                u[1:] = X[k+1:n, k]
            
            beta = 2.0 / u[0] if u[0] != 0.0 else 0.0
            
            temp = beta * torch.outer(u, torch.matmul(u, Q[k:n, :]))
            Q[k:n, :] -= temp
        
        return Q
    
    def apply_qt_to_vector(self, Q_implicit: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply Q' to vector."""
        return Q_implicit.T @ y
    
    def get_device_info(self) -> dict:
        """Get backend information."""
        return {
            'backend': 'gpu',
            'precision': 'fp64',
            'device': str(self.device),
            'library': f'PyTorch {self.torch.__version__}',
        }