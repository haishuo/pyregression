"""
Apple MPS (Metal Performance Shaders) GPU backend using ridge regression.

Uses PyTorch's MPS backend with ridge regression via Cholesky decomposition.
Ridge is used because QR decomposition is not available on Apple Metal.

This is a DIFFERENT algorithm than the CPU/CUDA backends:
- CPU/CUDA: QR decomposition (exact OLS)
- MPS: Ridge regression (regularized, approximate OLS)

Users should be aware of this distinction.
"""

import numpy as np
import warnings
import platform
from typing import Optional

from .base import GPUBackendFP32, LinearModelResult
from .ridge_suitability import check_ridge_suitability, format_suitability_message


class MPSRidgeBackendFP32(GPUBackendFP32):
    """
    Apple MPS backend using ridge regression.
    
    Algorithm: Ridge regression via Cholesky decomposition
    Hardware: Apple Silicon (M1/M2/M3/M4) only
    
    Uses PyTorch's Metal Performance Shaders (MPS) backend for GPU acceleration.
    Since QR decomposition is not available on Metal, this backend uses ridge
    regression as a numerically stable substitute for OLS.
    
    Requirements:
    - macOS 12.3+ on Apple Silicon
    - PyTorch with MPS support
    
    Precision: FP32
    Performance: ~10-50x speedup vs CPU (estimated)
    
    Important Notes:
    - NOT exact OLS (small ridge penalty λ added)
    - Not suitable for severely ill-conditioned problems (κ > 1e10)
    - Results differ slightly from QR-based OLS
    - For exact OLS, use CPU backend
    """
    
    def __init__(self):
        """Initialize MPS ridge backend."""
        self.name = "pytorch_mps_fp32_ridge"
        self.precision = "fp32"
        
        # Import PyTorch
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "MPS backend requires PyTorch. "
                "Install: pip install torch"
            )
        
        # Verify macOS
        if platform.system() != 'Darwin':
            raise RuntimeError(
                f"MPS backend requires macOS, detected: {platform.system()}\n"
                "For NVIDIA GPUs, use backend='pytorch'\n"
                "For CPU, use backend='cpu'"
            )
        
        # Verify MPS availability
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS not available. Requirements:\n"
                "  - macOS 12.3 or later\n"
                "  - Apple Silicon (M1/M2/M3/M4)\n"
                "  - PyTorch with MPS support\n"
                "Use backend='cpu' for CPU computation."
            )
        
        self.device = torch.device('mps')
    
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
        Fit linear model using ridge regression on MPS GPU.
        
        This method automatically checks if ridge regression can safely
        substitute QR decomposition for this problem. If the problem is
        severely ill-conditioned, an error is raised.
        
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
            Ignored (for API compatibility)
        singular_ok : bool
            If False, raise error for singular designs
        
        Returns
        -------
        LinearModelResult
            Fitted model results
        
        Raises
        ------
        ValueError
            If problem unsuitable for ridge regression
        """
        # Check if ridge regression is appropriate
        suitability = check_ridge_suitability(X, y, weights)
        
        if not suitability['suitable']:
            error_msg = format_suitability_message(suitability)
            raise ValueError(error_msg)
        
        # Warn user about ridge usage
        lam = suitability['recommended_lambda']
        
        warnings.warn(
            f"\nApple MPS GPU: Using ridge regression instead of QR.\n"
            f"{format_suitability_message(suitability)}",
            UserWarning,
            stacklevel=2
        )
        
        # Perform ridge regression
        return self._fit_ridge_cholesky(
            X=X,
            y=y,
            lam=lam,
            weights=weights,
            offset=offset,
            singular_ok=singular_ok
        )
    
    def _fit_ridge_cholesky(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lam: float,
        weights: Optional[np.ndarray],
        offset: Optional[np.ndarray],
        singular_ok: bool
    ) -> LinearModelResult:
        """
        Ridge regression via Cholesky decomposition on MPS GPU.
        
        Algorithm:
        1. Form X^T X + λI
        2. Cholesky: L L^T = X^T X + λI
        3. Solve L z = X^T y
        4. Solve L^T β = z
        """
        torch = self.torch
        n = len(y)
        
        # Convert to MPS tensors
        y_gpu = torch.from_numpy(y).float().to(self.device)
        X_full_gpu = torch.cat([
            torch.ones(n, 1, dtype=torch.float32, device=self.device),
            torch.from_numpy(X).float().to(self.device)
        ], dim=1)
        p = X_full_gpu.shape[1]
        
        # Handle offset
        if offset is not None:
            offset_gpu = torch.from_numpy(offset).float().to(self.device)
            y_work = y_gpu - offset_gpu
        else:
            y_work = y_gpu
        
        # Handle weights
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
        
        # Form Gram matrix: X^T X
        XtX = X_work.T @ X_work
        Xty = X_work.T @ y_work
        
        # Add ridge penalty: X^T X + λI
        XtX_ridge = XtX + float(lam) * torch.eye(p, dtype=torch.float32, device=self.device)
        
        # Cholesky decomposition (WORKS ON MPS!)
        try:
            L = torch.linalg.cholesky(XtX_ridge)
        except Exception as e:
            if not singular_ok:
                raise ValueError(f"Singular design matrix: {e}")
            # If Cholesky fails, increase lambda and retry
            lam_safe = lam * 100
            warnings.warn(
                f"Cholesky failed, increasing λ to {lam_safe:.2e}",
                UserWarning
            )
            XtX_ridge = XtX + float(lam_safe) * torch.eye(p, dtype=torch.float32, device=self.device)
            L = torch.linalg.cholesky(XtX_ridge)
        
        # Solve triangular systems
        # L z = X^T y
        z = torch.linalg.solve_triangular(L, Xty.unsqueeze(1), upper=False).squeeze(1)
        
        # L^T β = z
        coef = torch.linalg.solve_triangular(L.T, z.unsqueeze(1), upper=True).squeeze(1)
        
        # Compute fitted values and residuals
        fitted = X_full_gpu @ coef
        residuals = y_gpu - fitted
        
        # Adjust fitted for offset
        if offset is not None:
            fitted = fitted + offset_gpu
        
        # Ridge regression is always full rank
        rank = p
        
        # Convert results to numpy
        return LinearModelResult(
            coef=coef.cpu().numpy(),
            residuals=residuals.cpu().numpy(),
            fitted_values=fitted.cpu().numpy(),
            rank=rank,
            df_residual=n_good - rank,
            qr_R=L.cpu().numpy(),  # Store Cholesky factor (not QR's R)
            qr_pivot=np.arange(1, p + 1, dtype=np.int64),  # No pivoting
            qr_tol=float(lam)  # Store lambda (not tolerance)
        )
    
    def get_device_info(self) -> dict:
        """Get backend information."""
        try:
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=1
            )
            cpu_brand = result.stdout.strip()
        except:
            cpu_brand = platform.machine()
        
        return {
            'backend': 'gpu',
            'precision': 'fp32',
            'device': 'Apple MPS (Metal)',
            'chip': cpu_brand,
            'algorithm': 'Ridge regression via Cholesky',
            'library': f'PyTorch {self.torch.__version__} (MPS)',
            'note': 'Uses ridge instead of QR (QR not available on Metal)'
        }