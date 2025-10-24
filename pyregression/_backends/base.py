"""
Abstract base classes for backends.

Defines the interface all backends must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class QRResult:
    """Result from QR decomposition."""
    R: np.ndarray
    Q_implicit: np.ndarray
    pivot: np.ndarray
    rank: int
    tol: float


class BackendBase(ABC):
    """Abstract base class for all backends."""
    
    @abstractmethod
    def qr_with_pivoting(self, X: np.ndarray, tol: Optional[float] = None) -> QRResult:
        """QR decomposition with column pivoting."""
        pass
    
    @abstractmethod
    def apply_qt_to_vector(self, Q_implicit: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply Q' to vector y."""
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
        # TODO: Implement after precision detector
        from .cpu_fp64_backend import CPUBackendFP64
        return CPUBackendFP64()
