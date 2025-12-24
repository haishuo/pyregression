"""
Hardware precision capability detection for PyRegression.

Detects GPU hardware and determines optimal FP32/FP64 precision.
"""

import warnings
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class PrecisionSupport(Enum):
    """FP64 support level for hardware."""
    NO_GPU = "no_gpu"           # No GPU available
    NO_FP64 = "no_fp64"          # GPU exists but no FP64 (Apple Metal)
    GIMPED_FP64 = "gimped_fp64"  # FP64 exists but slow (consumer NVIDIA)
    FULL_FP64 = "full_fp64"      # Full-speed FP64 (A100, H100)


@dataclass
class GPUCapabilities:
    """
    GPU capability information.
    
    Attributes
    ----------
    has_gpu : bool
        Whether any GPU is available
    gpu_name : str
        Human-readable GPU name
    gpu_type : str
        Backend type: 'cuda', 'metal', or 'none'
    fp64_support : PrecisionSupport
        Level of FP64 support
    fp64_throughput_ratio : float
        Ratio of FP64 to FP32 throughput
    recommended_fp64 : bool
        Whether FP64 is recommended
    """
    has_gpu: bool
    gpu_name: str
    gpu_type: str
    fp64_support: PrecisionSupport
    fp64_throughput_ratio: float
    recommended_fp64: bool


def detect_gpu_capabilities() -> GPUCapabilities:
    """
    Detect GPU hardware and FP64 capabilities.
    
    Returns
    -------
    GPUCapabilities
        Detected hardware capabilities
    """
    # Try CUDA first
    cuda_caps = _detect_cuda_capabilities()
    if cuda_caps is not None:
        return cuda_caps
    
    # Try Metal second
    metal_caps = _detect_metal_capabilities()
    if metal_caps is not None:
        return metal_caps
    
    # No GPU available
    return GPUCapabilities(
        has_gpu=False,
        gpu_name="CPU only",
        gpu_type="none",
        fp64_support=PrecisionSupport.NO_GPU,
        fp64_throughput_ratio=1.0,
        recommended_fp64=True  # CPU always supports FP64
    )


def _detect_cuda_capabilities() -> Optional[GPUCapabilities]:
    """Detect NVIDIA CUDA GPU capabilities."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
        
        # Get GPU name
        gpu_name = torch.cuda.get_device_name(0)
        
        # Classify GPU
        support, ratio, recommended = _classify_nvidia_gpu(gpu_name)
        
        return GPUCapabilities(
            has_gpu=True,
            gpu_name=gpu_name,
            gpu_type="cuda",
            fp64_support=support,
            fp64_throughput_ratio=ratio,
            recommended_fp64=recommended
        )
        
    except ImportError:
        return None


def _detect_metal_capabilities() -> Optional[GPUCapabilities]:
    """Detect Apple Metal GPU capabilities."""
    try:
        import torch
        
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            return None
        
        # Metal doesn't support FP64
        return GPUCapabilities(
            has_gpu=True,
            gpu_name="Apple Metal GPU",
            gpu_type="metal",
            fp64_support=PrecisionSupport.NO_FP64,
            fp64_throughput_ratio=0.0,
            recommended_fp64=False
        )
        
    except ImportError:
        return None


def _classify_nvidia_gpu(gpu_name: str) -> tuple[PrecisionSupport, float, bool]:
    """
    Classify NVIDIA GPU FP64 capabilities.
    
    Parameters
    ----------
    gpu_name : str
        GPU name from torch.cuda.get_device_name()
        
    Returns
    -------
    (support_level, throughput_ratio, recommended)
    """
    gpu_upper = gpu_name.upper()
    
    # Data center GPUs with full FP64
    full_fp64_models = [
        'A100', 'A800',
        'H100', 'H800',
        'V100',
        'P100',
    ]
    
    for model in full_fp64_models:
        if model in gpu_upper:
            return PrecisionSupport.FULL_FP64, 0.5, True
    
    # RTX 50 series (Blackwell) - 1/64 ratio
    if 'RTX 50' in gpu_upper:
        return PrecisionSupport.GIMPED_FP64, 1/64, False
    
    # RTX 40 series (Ada Lovelace) - 1/64 ratio
    if 'RTX 40' in gpu_upper:
        return PrecisionSupport.GIMPED_FP64, 1/64, False
    
    # RTX 30 series (Ampere) - 1/64 ratio
    if 'RTX 30' in gpu_upper:
        return PrecisionSupport.GIMPED_FP64, 1/64, False
    
    # RTX 20 series (Turing) - 1/32 ratio
    if 'RTX 20' in gpu_upper:
        return PrecisionSupport.GIMPED_FP64, 1/32, False
    
    # GTX series - 1/32 ratio
    if 'GTX' in gpu_upper:
        return PrecisionSupport.GIMPED_FP64, 1/32, False
    
    # Unknown - assume gimped
    warnings.warn(
        f"Unknown NVIDIA GPU '{gpu_name}'. Assuming gimped FP64."
    )
    return PrecisionSupport.GIMPED_FP64, 1/32, False


def validate_fp64_request(capabilities: GPUCapabilities, use_fp64: bool) -> None:
    """
    Validate user's FP64 request against hardware.
    
    Raises
    ------
    RuntimeError
        If FP64 requested on Metal (no support)
    
    Warns
    -----
    UserWarning
        If FP64 requested on gimped hardware
    """
    if not use_fp64:
        return  # FP32 always works
    
    if capabilities.fp64_support == PrecisionSupport.NO_FP64:
        raise RuntimeError(
            f"FP64 requested but not supported on {capabilities.gpu_name}. "
            f"Apple Metal does not support FP64. "
            f"Use FP32 (use_fp64=False) or CPU backend."
        )
    
    if capabilities.fp64_support == PrecisionSupport.GIMPED_FP64:
        warnings.warn(
            f"FP64 requested on {capabilities.gpu_name} with gimped FP64 "
            f"(ratio: {capabilities.fp64_throughput_ratio:.3f}). "
            f"This will be ~{int(1/capabilities.fp64_throughput_ratio)}x slower than FP32. "
            f"Consider FP32 for better performance.",
            UserWarning
        )


def recommend_precision(capabilities: GPUCapabilities,
                        user_preference: Optional[bool]) -> bool:
    """
    Recommend FP64 vs FP32 based on hardware.
    
    Parameters
    ----------
    capabilities : GPUCapabilities
        Detected hardware
    user_preference : Optional[bool]
        User's preference (None for auto)
        
    Returns
    -------
    bool
        True for FP64, False for FP32
    """
    if user_preference is not None:
        validate_fp64_request(capabilities, user_preference)
        return user_preference
    
    return capabilities.recommended_fp64


def print_capabilities() -> None:
    """Print detected GPU capabilities (for debugging)."""
    caps = detect_gpu_capabilities()
    
    print("GPU Capability Detection")
    print("=" * 50)
    print(f"GPU Available: {caps.has_gpu}")
    print(f"GPU Name: {caps.gpu_name}")
    print(f"GPU Type: {caps.gpu_type}")
    print(f"FP64 Support: {caps.fp64_support.value}")
    print(f"FP64/FP32 Ratio: {caps.fp64_throughput_ratio:.4f}")
    print(f"Recommended FP64: {caps.recommended_fp64}")
    
    if caps.fp64_support == PrecisionSupport.GIMPED_FP64:
        print(f"⚠️  FP64 is ~{int(1/caps.fp64_throughput_ratio)}x slower than FP32")
    elif caps.fp64_support == PrecisionSupport.NO_FP64:
        print("❌ No FP64 support on this GPU")
    elif caps.fp64_support == PrecisionSupport.FULL_FP64:
        print("✅ Full-speed FP64 available")


if __name__ == "__main__":
    print_capabilities()