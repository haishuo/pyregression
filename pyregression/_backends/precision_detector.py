"""
GPU capability detection and precision support analysis.

Detects NVIDIA CUDA and Apple Metal GPUs.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import platform


class PrecisionSupport(Enum):
    """GPU FP64 support levels."""
    FULL_FP64 = "full_fp64"      # Full speed FP64 (A100, H100)
    GIMPED_FP64 = "gimped_fp64"  # Throttled FP64 (RTX, consumer cards)
    NO_FP64 = "no_fp64"          # FP32 only (Apple Silicon)


@dataclass
class GPUCapabilities:
    """GPU hardware capabilities."""
    has_gpu: bool
    gpu_name: str
    gpu_type: str  # 'nvidia', 'mlx', or 'none'
    fp64_support: PrecisionSupport
    fp64_throughput_ratio: float
    recommended_fp64: bool


def _detect_nvidia_capabilities() -> Optional[GPUCapabilities]:
    """Detect NVIDIA GPU via PyTorch."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        major, minor = props.major, props.minor
        
        # Determine FP64 support
        # Professional GPUs: Full speed FP64
        is_professional = any(
            prefix in gpu_name.upper()
            for prefix in ['TESLA', 'A100', 'H100', 'A40', 'V100', 'P100', 'QUADRO']
        )
        
        if is_professional:
            return GPUCapabilities(
                has_gpu=True,
                gpu_name=f"{gpu_name} (SM {major}.{minor})",
                gpu_type="nvidia",
                fp64_support=PrecisionSupport.FULL_FP64,
                fp64_throughput_ratio=0.5,
                recommended_fp64=True
            )
        else:
            # Consumer GPUs: 1/32 FP64 throughput
            return GPUCapabilities(
                has_gpu=True,
                gpu_name=f"{gpu_name} (SM {major}.{minor})",
                gpu_type="nvidia",
                fp64_support=PrecisionSupport.GIMPED_FP64,
                fp64_throughput_ratio=1/32,
                recommended_fp64=False
            )
    
    except ImportError:
        return None
    except Exception:
        return None


def _detect_mlx_capabilities() -> Optional[GPUCapabilities]:
    """Detect Apple Silicon via MLX."""
    try:
        import mlx.core as mx
        
        # Verify macOS
        if platform.system() != 'Darwin':
            return None
        
        # Verify Apple Silicon
        machine = platform.machine()
        if machine not in ['arm64', 'aarch64']:
            return None
        
        # Try to get chip name
        try:
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=1
            )
            cpu_brand = result.stdout.strip()
            if 'Apple' in cpu_brand:
                gpu_name = f"Apple Silicon GPU ({cpu_brand})"
            else:
                gpu_name = f"Apple Silicon GPU ({machine})"
        except:
            gpu_name = f"Apple Silicon GPU ({machine})"
        
        return GPUCapabilities(
            has_gpu=True,
            gpu_name=gpu_name,
            gpu_type="mlx",
            fp64_support=PrecisionSupport.NO_FP64,
            fp64_throughput_ratio=0.0,
            recommended_fp64=False
        )
    
    except ImportError:
        return None
    except Exception:
        return None


def detect_gpu_capabilities() -> GPUCapabilities:
    """Detect available GPU and capabilities.
    
    Priority: NVIDIA CUDA → Apple MLX → No GPU
    """
    # Try NVIDIA first
    nvidia = _detect_nvidia_capabilities()
    if nvidia:
        return nvidia
    
    # Try Apple Silicon
    mlx = _detect_mlx_capabilities()
    if mlx:
        return mlx
    
    # No GPU
    return GPUCapabilities(
        has_gpu=False,
        gpu_name="No GPU detected (CPU only)",
        gpu_type="none",
        fp64_support=PrecisionSupport.NO_FP64,
        fp64_throughput_ratio=0.0,
        recommended_fp64=False
    )


def recommend_precision(caps: GPUCapabilities, use_fp64: Optional[bool]) -> bool:
    """Recommend precision based on hardware.
    
    Returns True if FP64 should be used.
    """
    # User explicitly requested FP64
    if use_fp64 is True:
        return True
    
    # User explicitly requested FP32
    if use_fp64 is False:
        return False
    
    # Auto-detect
    if not caps.has_gpu:
        return True  # CPU always uses FP64
    
    if caps.gpu_type == 'mlx':
        return False  # Apple Silicon only has FP32
    
    # NVIDIA: use FP64 only if professional GPU
    return caps.recommended_fp64


def validate_fp64_request(caps: GPUCapabilities, use_fp64: bool):
    """Validate FP64 request against hardware capabilities."""
    if use_fp64 and caps.has_gpu and caps.fp64_support == PrecisionSupport.NO_FP64:
        raise ValueError(
            f"FP64 requested but {caps.gpu_name} does not support FP64. "
            "Please use FP32 (use_fp64=False) or CPU backend."
        )


def print_gpu_info():
    """Print GPU detection results (diagnostic)."""
    caps = detect_gpu_capabilities()
    
    print("GPU Detection Results:")
    print(f"  GPU Available: {caps.has_gpu}")
    print(f"  GPU Name: {caps.gpu_name}")
    print(f"  GPU Type: {caps.gpu_type}")
    print(f"  FP64 Support: {caps.fp64_support.value}")
    print(f"  FP64/FP32 Ratio: {caps.fp64_throughput_ratio:.3f}")
    print(f"  Recommended FP64: {caps.recommended_fp64}")
    
    use_fp64 = recommend_precision(caps, None)
    print(f"\nRecommended Precision: {'FP64' if use_fp64 else 'FP32'}")


if __name__ == "__main__":
    print_gpu_info()