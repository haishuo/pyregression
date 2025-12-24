"""
GPU capability detection for precision selection.

Detects NVIDIA GPUs (via PyTorch CUDA) and Apple Silicon (via PyTorch MPS).
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class FP64Support(Enum):
    """FP64 support level."""
    NATIVE = "native"      # Full-speed FP64 (professional GPUs)
    SLOW = "slow"          # Emulated FP64 (1/32 speed on consumer GPUs)
    NONE = "none"          # No FP64 support


@dataclass
class GPUCapabilities:
    """GPU hardware capabilities."""
    has_gpu: bool
    gpu_type: str  # 'nvidia', 'mps', or 'none'
    gpu_name: str
    fp64_support: FP64Support
    recommended_fp64: bool


def detect_gpu_capabilities() -> GPUCapabilities:
    """
    Detect GPU capabilities for backend selection.
    
    Returns
    -------
    GPUCapabilities
        Detected GPU information
    
    Examples
    --------
    >>> caps = detect_gpu_capabilities()
    >>> if caps.has_gpu and caps.gpu_type == 'nvidia':
    ...     print(f"Found NVIDIA GPU: {caps.gpu_name}")
    >>> if caps.recommended_fp64:
    ...     print("FP64 recommended for this hardware")
    """
    # Check for NVIDIA GPU via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            
            # Detect FP64 support based on compute capability
            props = torch.cuda.get_device_properties(0)
            major = props.major
            minor = props.minor
            
            # Professional GPUs (compute capability >= 6.0) have native FP64
            # Consumer GPUs have 1/32 speed FP64
            if major >= 6:
                fp64_support = FP64Support.NATIVE
            elif major == 5 or (major == 3 and minor >= 5):
                fp64_support = FP64Support.SLOW
            else:
                fp64_support = FP64Support.NONE
            
            return GPUCapabilities(
                has_gpu=True,
                gpu_type='nvidia',
                gpu_name=gpu_name,
                fp64_support=fp64_support,
                recommended_fp64=(fp64_support == FP64Support.NATIVE)
            )
    except ImportError:
        pass
    
    # Check for Apple Silicon MPS
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            import platform
            gpu_name = f"Apple {platform.processor()} (MPS)"
            
            return GPUCapabilities(
                has_gpu=True,
                gpu_type='mps',
                gpu_name=gpu_name,
                fp64_support=FP64Support.NONE,
                recommended_fp64=False
            )
    except (ImportError, AttributeError):
        pass
    
    # No GPU detected - CPU only
    return GPUCapabilities(
        has_gpu=False,
        gpu_type='none',
        gpu_name='CPU',
        fp64_support=FP64Support.NATIVE,
        recommended_fp64=True
    )


def recommend_precision(caps: GPUCapabilities, user_preference: Optional[bool]) -> bool:
    """
    Recommend FP32 vs FP64 based on hardware and user preference.
    
    Parameters
    ----------
    caps : GPUCapabilities
        Hardware capabilities
    user_preference : bool or None
        User's precision preference (None = auto-detect)
    
    Returns
    -------
    bool
        True if FP64 is recommended, False for FP32
    
    Decision logic:
    - User explicitly requests FP64 → FP64 (may be slow on consumer GPU)
    - User explicitly requests FP32 → FP32
    - No user preference:
        - Professional GPU (native FP64) → FP64
        - Consumer GPU (slow FP64) → FP32
        - Apple Silicon (no FP64) → FP32
        - CPU → FP64
    
    Examples
    --------
    >>> caps = detect_gpu_capabilities()
    >>> use_fp64 = recommend_precision(caps, user_preference=None)
    """
    if user_preference is not None:
        return user_preference
    
    # Auto-detect based on hardware
    return caps.recommended_fp64