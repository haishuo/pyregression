"""
Backend selection and management.

Provides unified interface for CPU, NVIDIA GPU, and Apple Silicon GPU backends.
"""

from typing import Optional
import warnings

from .base import BackendBase, BackendFactory
from .precision_detector import (
    detect_gpu_capabilities,
    recommend_precision,
    GPUCapabilities
)
from .ridge_suitability import check_ridge_suitability

# Try importing CPU backend (always available)
try:
    from .cpu_fp64_backend import CPUBackendFP64
    CPU_AVAILABLE = True
except ImportError:
    CPU_AVAILABLE = False
    warnings.warn("CPU backend unavailable - installation error!")

# Try importing PyTorch backends
try:
    from .gpu_fp32_backend import PyTorchBackendFP32
    PYTORCH_FP32_AVAILABLE = True
except ImportError:
    PYTORCH_FP32_AVAILABLE = False

try:
    from .gpu_fp64_backend import PyTorchBackendFP64
    PYTORCH_FP64_AVAILABLE = True
except ImportError:
    PYTORCH_FP64_AVAILABLE = False

# Try importing MPS ridge backend
try:
    from .gpu_mps_ridge_backend import MPSRidgeBackendFP32
    MPS_AVAILABLE = True
except ImportError:
    MPS_AVAILABLE = False


def get_backend(backend: str = 'auto', use_fp64: Optional[bool] = None) -> BackendBase:
    """
    Get computational backend.
    
    Parameters
    ----------
    backend : str
        Backend selection:
        - 'auto': Auto-select based on hardware
        - 'cpu': CPU with NumPy (FP64, R-compatible)
        - 'gpu': Any available GPU (PyTorch or MLX)
        - 'pytorch': Force PyTorch CUDA (NVIDIA only)
        - 'mlx': Force MLX Metal (Apple Silicon only)
    
    use_fp64 : bool or None
        Precision preference:
        - None: Auto-detect
        - True: Force FP64 (CPU or professional GPU)
        - False: Allow FP32 (GPU)
    
    Returns
    -------
    BackendBase
        Backend instance
    
    Examples
    --------
    >>> # Auto-select best backend
    >>> backend = get_backend('auto')
    
    >>> # Force CPU for regulatory compliance
    >>> backend = get_backend('cpu')
    
    >>> # Use any GPU
    >>> backend = get_backend('gpu')
    
    >>> # Force specific GPU type
    >>> backend = get_backend('pytorch')  # NVIDIA
    >>> backend = get_backend('mlx')      # Apple
    """
    
    if backend == 'auto':
        caps = detect_gpu_capabilities()
        use_fp64_final = recommend_precision(caps, use_fp64)
        
        # If FP64 required, use CPU or professional GPU
        if use_fp64_final:
            if caps.has_gpu and caps.recommended_fp64 and PYTORCH_FP64_AVAILABLE:
                return PyTorchBackendFP64()
            else:
                if not CPU_AVAILABLE:
                    raise RuntimeError("CPU backend unavailable!")
                return CPUBackendFP64()
        
        # FP32 allowed - use GPU if available
        if caps.has_gpu:
            if caps.gpu_type == 'nvidia' and PYTORCH_FP32_AVAILABLE:
                return PyTorchBackendFP32()
            elif caps.gpu_type == 'mlx' and MLX_AVAILABLE:
                return MLXBackendFP32()
        
        # Default to CPU
        if not CPU_AVAILABLE:
            raise RuntimeError("No backends available!")
        return CPUBackendFP64()
    
    elif backend == 'cpu':
        if not CPU_AVAILABLE:
            raise RuntimeError("CPU backend unavailable!")
        return CPUBackendFP64()
    
    elif backend == 'gpu':
        caps = detect_gpu_capabilities()
        
        if not caps.has_gpu:
            raise ValueError(
                "No GPU detected.\n"
                "Options:\n"
                "  - Use backend='cpu'\n"
                "  - Install PyTorch with CUDA for NVIDIA\n"
                "  - Install MLX for Apple Silicon"
            )
        
        # Route to appropriate GPU
        if caps.gpu_type == 'nvidia':
            # Choose FP32 or FP64 based on use_fp64
            use_fp64_final = recommend_precision(caps, use_fp64)
            if use_fp64_final and PYTORCH_FP64_AVAILABLE:
                return PyTorchBackendFP64()
            elif PYTORCH_FP32_AVAILABLE:
                return PyTorchBackendFP32()
            else:
                raise RuntimeError(
                    "NVIDIA GPU detected but PyTorch unavailable.\n"
                    "Install: pip install torch"
                )
        
        elif caps.gpu_type == 'mlx':
            if not MLX_AVAILABLE:
                raise RuntimeError(
                    "Apple Silicon detected but MLX unavailable.\n"
                    "Install: pip install mlx"
                )
            return MLXBackendFP32()
        
        else:
            raise RuntimeError(f"Unsupported GPU type: {caps.gpu_type}")
    
    elif backend == 'pytorch':
        # Force PyTorch (auto-select FP32 vs FP64)
        caps = detect_gpu_capabilities()
        use_fp64_final = recommend_precision(caps, use_fp64)
        
        if use_fp64_final and PYTORCH_FP64_AVAILABLE:
            return PyTorchBackendFP64()
        elif PYTORCH_FP32_AVAILABLE:
            return PyTorchBackendFP32()
        else:
            raise RuntimeError(
                "PyTorch backend unavailable.\n"
                "Install: pip install torch"
            )
    
    elif backend == 'mlx':
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX backend unavailable.\n"
                "Install: pip install mlx\n"
                "Note: MLX only works on macOS with Apple Silicon"
            )
        return MLXBackendFP32()
    
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'\n"
            f"Valid options: 'auto', 'cpu', 'gpu', 'pytorch', 'mlx'"
        )


def list_available_backends() -> list:
    """List names of available backends."""
    backends = []
    if CPU_AVAILABLE:
        backends.append('cpu')
    if PYTORCH_FP32_AVAILABLE or PYTORCH_FP64_AVAILABLE:
        backends.append('pytorch')
    if MPS_AVAILABLE:
        backends.append('mps')
    return backends


def print_backend_info():
    """Print detailed backend information (diagnostic)."""
    caps = detect_gpu_capabilities()
    
    print("PyRegression Backend Status")
    print("=" * 50)
    print(f"\nAvailable Backends:")
    print(f"  CPU (FP64):          {'✓' if CPU_AVAILABLE else '✗'} - QR decomposition (exact OLS)")
    print(f"  PyTorch CUDA (FP32): {'✓' if PYTORCH_FP32_AVAILABLE else '✗'} - QR decomposition (exact OLS)")
    print(f"  PyTorch CUDA (FP64): {'✓' if PYTORCH_FP64_AVAILABLE else '✗'} - QR decomposition (exact OLS)")
    print(f"  MPS Ridge (FP32):    {'✓' if MPS_AVAILABLE else '✗'} - Ridge regression (approximate OLS)")
    
    print(f"\nHardware Detection:")
    if caps.has_gpu:
        print(f"  GPU Type: {caps.gpu_type}")
        print(f"  GPU Name: {caps.gpu_name}")
        print(f"  FP64 Support: {caps.fp64_support.value}")
    else:
        print(f"  No GPU detected")
    
    print(f"\nRecommended Backend:")
    try:
        backend = get_backend('auto')
        print(f"  {backend.name}")
        if 'ridge' in backend.name:
            print(f"  Note: Uses ridge regression (not exact OLS)")
    except Exception as e:
        print(f"  Error: {e}")


# Export main interface
__all__ = [
    'get_backend',
    'list_available_backends',
    'print_backend_info',
    'BackendBase',
    'BackendFactory',
    'detect_gpu_capabilities',
    'check_ridge_suitability',  # For advanced users
]


if __name__ == "__main__":
    print_backend_info()