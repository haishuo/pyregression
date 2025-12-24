"""
GPU backend using PyTorch with FP32 precision.

NVIDIA CUDA GPUs only. For Apple Silicon, use MLX backend.
"""

import numpy as np
import warnings
from typing import Optional, Any

from .base import GPUBackendFP32, LinearModelResult


class PyTorchBackendFP32(GPUBackendFP32):
    """
    PyTorch GPU backend with FP32 precision.
    
    Keeps all computation on GPU using torch tensors.
    Only converts at entry (numpy → torch) and exit (torch → numpy).
    
    Requirements:
    - NVIDIA GPU with CUDA support
    - PyTorch with CUDA enabled
    
    For Apple Silicon GPUs, use MLXBackendFP32 instead.
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize PyTorch FP32 backend."""
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
        """Select CUDA GPU device. Fails if CUDA unavailable."""
        torch = self.torch
        
        if requested:
            device = torch.device(requested)
            
            # Explicitly reject MPS
            if device.type == 'mps':
                raise ValueError(
                    "PyTorch backend does not support Apple MPS (Metal). "
                    "For Apple Silicon GPU acceleration, use: "
                    "backend = get_backend('mlx')"
                )
            
            return device
        
        # Auto-detect CUDA only
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch backend requires NVIDIA CUDA GPU.\n"
                "Options:\n"
                "  1. Use get_backend('cpu') for CPU (FP64)\n"
                "  2. Use get_backend('mlx') for Apple Silicon GPU\n"
                "  3. Install CUDA-enabled PyTorch"
            )
        
        return torch.device('cuda')
    
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
        Fit linear model on GPU.
        
        ALL computation happens on GPU with torch tensors.
        Only convert at boundaries (entry/exit).
        """
        torch = self.torch
        n = len(y)
        
        # Convert to GPU tensors ONCE at entry
        y_gpu = torch.from_numpy(y).float().to(self.device)
        X_full_gpu = torch.cat([
            torch.ones(n, 1, dtype=torch.float32, device=self.device),
            torch.from_numpy(X).float().to(self.device)
        ], dim=1)
        p = X_full_gpu.shape[1]
        
        # Adjust for offset (on GPU)
        if offset is not None:
            offset_gpu = torch.from_numpy(offset).float().to(self.device)
            y_work = y_gpu - offset_gpu
        else:
            y_work = y_gpu.clone()
        
        # Handle weights (on GPU)
        if weights is not None:
            weights_gpu = torch.from_numpy(weights).float().to(self.device)
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
        
        # Compute tolerance (on GPU)
        if tol is None:
            # Use a more conservative tolerance for FP32
            # Scale by number of columns (p), not rows (n)
            # This prevents tolerance explosion on large matrices
            eps = torch.finfo(torch.float32).eps
            norm_X = torch.linalg.norm(X_work, 'fro')
            tol = p * eps * norm_X  # Use p, not max(n_good, p)
            tol = float(tol.item())
        
        # QR decomposition with pivoting (on GPU)
        Q, R, pivot = self._qr_with_pivoting_gpu(X_work, tol)
        
        # Determine rank (on GPU)
        R_diag = torch.abs(torch.diag(R))
        if R_diag[0] == 0:
            rank = 0
        else:
            rank = int(torch.sum(R_diag >= tol * R_diag[0]).item())
        
        if not singular_ok and rank < p:
            raise ValueError(f"Singular fit: rank {rank} < {p} columns")
        
        # Solve R β = Q'y (on GPU)
        qty = Q.T @ y_work
        
        # Initialize coefficients (on GPU)
        coef = torch.full((p,), float('nan'), dtype=torch.float32, device=self.device)
        
        if rank > 0:
            # Back-solve (on GPU)
            coef_active = torch.linalg.solve_triangular(
                R[:rank, :rank],
                qty[:rank].unsqueeze(1),  # Make it (rank, 1) instead of (rank,)
                upper=True
            ).squeeze(1)  # Then squeeze back to (rank,)
            coef[pivot[:rank]] = coef_active
        
        # Compute fitted values (on GPU, handling NaN coefficients)
        valid_coef = ~torch.isnan(coef)
        if torch.any(valid_coef):
            fitted = X_full_gpu[:, valid_coef] @ coef[valid_coef]
        else:
            fitted = torch.zeros(n, dtype=torch.float32, device=self.device)
        
        residuals = y_gpu - fitted
        
        # Adjust fitted for offset (on GPU)
        if offset is not None:
            fitted = fitted + offset_gpu
        
        # Convert ONCE at exit
        return LinearModelResult(
            coef=coef.cpu().numpy(),
            residuals=residuals.cpu().numpy(),
            fitted_values=fitted.cpu().numpy(),
            rank=rank,
            df_residual=n_good - rank,
            qr_R=R[:p, :p].cpu().numpy(),
            qr_pivot=(pivot.cpu().numpy() + 1).astype(np.int64),  # 1-indexed
            qr_tol=tol
        )
    
    def _qr_with_pivoting_gpu(self, X, tol):
        """
        QR with column pivoting on GPU.
        
        Simplified version - just use unpivoted QR for now.
        TODO: Implement proper Householder with pivoting.
        """
        torch = self.torch
        n, p = X.shape
        
        # For now: unpivoted QR + identity pivot
        # This works but isn't optimal for rank-deficient cases
        Q, R = torch.linalg.qr(X, mode='complete')
        pivot = torch.arange(p, dtype=torch.int64, device=self.device)
        
        return Q, R, pivot
    
    def get_device_info(self) -> dict:
        """Get backend information."""
        return {
            'backend': 'gpu',
            'precision': 'fp32',
            'device': str(self.device),
            'library': f'PyTorch {self.torch.__version__}',
        }