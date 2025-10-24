"""
Backend module for PyRegression.

Provides unified interface for CPU and GPU computation.
"""

from .base import (
    BackendBase,
    CPUBackend,
    GPUBackendFP32,
    GPUBackendFP64,
    BackendFactory,
)

# Try to import concrete backends
try:
    from .cpu_fp64_backend import CPUBackendFP64
    CPU_AVAILABLE = True
except ImportError:
    CPU_AVAILABLE = False

try:
    from .gpu_fp32_backend import GPUBackendFP32 as GPUBackendFP32Impl
    from .gpu_fp64_backend import GPUBackendFP64 as GPUBackendFP64Impl
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def get_backend(backend: str = 'auto', use_fp64=None):
    """
    Get computational backend.
    
    Parameters
    ----------
    backend : str
        'auto', 'cpu', 'gpu'
    use_fp64 : bool, optional
        Use FP64 precision
        
    Returns
    -------
    backend : BackendBase
        Backend instance
    """
    if backend == 'cpu' or backend == 'auto':
        if not CPU_AVAILABLE:
            raise ImportError("CPU backend not available")
        return CPUBackendFP64()
    elif backend == 'gpu':
        if not GPU_AVAILABLE:
            raise ImportError(
                "GPU backend requires PyTorch. "
                "Install with: pip install torch"
            )
        return BackendFactory.get_optimal_backend(use_fp64=use_fp64)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def list_available_backends():
    """List available backends."""
    backends = []
    if CPU_AVAILABLE:
        backends.append('cpu')
    if GPU_AVAILABLE:
        backends.append('gpu')
    return backends


__all__ = [
    'BackendBase',
    'CPUBackend',
    'GPUBackendFP32',
    'GPUBackendFP64',
    'BackendFactory',
    'get_backend',
    'list_available_backends',
]
