"""
GPU backend using PyTorch with FP64 precision.

For data center GPUs: A100, H100, V100.
"""

import numpy as np
import warnings
from typing import Optional

from .gpu_fp32_backend import PyTorchBackendFP32
from .base import GPUBackendFP64, LinearModelResult


class PyTorchBackendFP64(GPUBackendFP64):
    """
    PyTorch GPU backend with FP64 precision.
    
    Same as FP32 but uses float64 precision.
    Only recommended for data center GPUs with full FP64 support.
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize PyTorch FP64 backend."""
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
        
        # Device selection (no Metal for FP64)
        if device == 'mps':
            raise RuntimeError(
                "FP64 not supported on Apple Metal. "
                "Use FP32 backend or CPU."
            )
        
        if device is None:
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
                    f"This will be ~{int(1/caps.fp64_throughput_ratio)}x slower than FP32.",
                    UserWarning
                )
    
    def _select_device(self, requested):
        """Not used - device selection in __init__."""
        return self.device
    
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
        Fit linear model on GPU with FP64 precision.
        
        Same algorithm as FP32 but uses double precision.
        """
        torch = self.torch
        n = len(y)
        
        # Convert to GPU tensors with FP64
        y_gpu = torch.from_numpy(y).double().to(self.device)
        X_full_gpu = torch.cat([
            torch.ones(n, 1, dtype=torch.float64, device=self.device),
            torch.from_numpy(X).double().to(self.device)
        ], dim=1)
        p = X_full_gpu.shape[1]
        
        # Adjust for offset
        if offset is not None:
            offset_gpu = torch.from_numpy(offset).double().to(self.device)
            y_work = y_gpu - offset_gpu
        else:
            y_work = y_gpu.clone()
        
        # Handle weights
        if weights is not None:
            weights_gpu = torch.from_numpy(weights).double().to(self.device)
            good = weights_gpu > 0
            n_good = int(torch.sum(good).item())
            
            if n_good == 0:
                raise ValueError("All weights are zero")
            
            w_sqrt = torch.sqrt(weights_gpu[good])
            X_work = X_full_gpu[good, :] * w_sqrt.unsqueeze(1)
            y_work = y_work[good] * w_sqrt
        else:
            X_work = X_full_gpu
            n_good = n
        
        # Compute tolerance
        if tol is None:
            # For FP64, we can use the standard formula
            # But still prefer p over n for consistency
            eps = torch.finfo(torch.float64).eps
            norm_X = torch.linalg.norm(X_work, 'fro')
            tol = p * eps * norm_X  # Use p for consistency with FP32
            tol = float(tol.item())
        
        # QR decomposition with pivoting
        Q, R, pivot = self._qr_with_pivoting_gpu(X_work, tol)
        
        # Determine rank
        R_diag = torch.abs(torch.diag(R))
        if R_diag[0] == 0:
            rank = 0
        else:
            rank = int(torch.sum(R_diag >= tol * R_diag[0]).item())
        
        if not singular_ok and rank < p:
            raise ValueError(f"Singular fit: rank {rank} < {p} columns")
        
        # Solve R β = Q'y
        qty = Q.T @ y_work
        
        # Initialize coefficients
        coef = torch.full((p,), float('nan'), dtype=torch.float64, device=self.device)
        
        if rank > 0:
            # Back-solve
            coef_active = torch.linalg.solve_triangular(
                R[:rank, :rank],
                qty[:rank].unsqueeze(1),  # Make it (rank, 1)
                upper=True
            ).squeeze(1)  # Squeeze back to (rank,)
            coef[pivot[:rank]] = coef_active
        
        # Compute fitted values (handling NaN coefficients)
        valid_coef = ~torch.isnan(coef)
        if torch.any(valid_coef):
            fitted = X_full_gpu[:, valid_coef] @ coef[valid_coef]
        else:
            fitted = torch.zeros(n, dtype=torch.float64, device=self.device)
        
        residuals = y_gpu - fitted
        
        # Adjust fitted for offset
        if offset is not None:
            fitted = fitted + offset_gpu
        
        # Convert to numpy
        return LinearModelResult(
            coef=coef.cpu().numpy(),
            residuals=residuals.cpu().numpy(),
            fitted_values=fitted.cpu().numpy(),
            rank=rank,
            df_residual=n_good - rank,
            qr_R=R[:p, :p].cpu().numpy(),
            qr_pivot=(pivot.cpu().numpy() + 1).astype(np.int64),
            qr_tol=tol
        )
    
    def _qr_with_pivoting_gpu(self, X, tol):
        """QR with pivoting - simplified for now."""
        torch = self.torch
        p = X.shape[1]
        
        # Unpivoted QR
        Q, R = torch.linalg.qr(X, mode='complete')
        pivot = torch.arange(p, dtype=torch.int64, device=self.device)
        
        return Q, R, pivot
    
    def get_device_info(self) -> dict:
        """Get backend information."""
        return {
            'backend': 'gpu',
            'precision': 'fp64',
            'device': str(self.device),
            'library': f'PyTorch {self.torch.__version__}',
        }