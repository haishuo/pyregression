"""
Abstract base classes for backends.

Defines the interface all backends must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class LinearModelResult:
    """Complete linear regression results."""
    coef: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray
    rank: int
    df_residual: int
    qr_R: np.ndarray
    qr_pivot: np.ndarray
    qr_tol: float


class BackendBase(ABC):
    """Abstract base class for all backends."""
    
    @abstractmethod
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
        Fit linear model - complete computation.
        
        Backends implement ALL computation internally using their
        native types, only converting at entry/exit.
        
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
            Tolerance for rank determination
        singular_ok : bool
            Allow singular fits
            
        Returns
        -------
        LinearModelResult
            Complete regression results (all numpy arrays)
        """
        pass
    
    @abstractmethod
    def get_device_info(self) -> dict:
        """Get backend information."""
        pass


class CPUBackend(BackendBase):
    """CPU backend base class (always FP64)."""
    pass


class GPUBackendFP32(BackendBase):
    """GPU backend base class for FP32."""
    pass


class GPUBackendFP64(BackendBase):
    """GPU backend base class for FP64."""
    pass


class BackendFactory:
    """Factory for creating backends."""
    
    @staticmethod
    def get_optimal_backend(use_fp64: Optional[bool] = None):
        """Get optimal backend based on hardware."""
        from .precision_detector import detect_gpu_capabilities, recommend_precision
        
        caps = detect_gpu_capabilities()
        
        if use_fp64 is None:
            use_fp64 = recommend_precision(caps, None)
        
        if not caps.has_gpu:
            from .cpu_fp64_backend import CPUBackendFP64
            return CPUBackendFP64()
        
        # GPU available
        if use_fp64:
            from .gpu_fp64_backend import PyTorchBackendFP64
            return PyTorchBackendFP64()
        else:
            from .gpu_fp32_backend import PyTorchBackendFP32
            return PyTorchBackendFP32()