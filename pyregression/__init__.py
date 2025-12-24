"""
PyRegression: GPU-accelerated statistical inference with R-compatible numerics.

Copyright (C) 2024 SGCX
Licensed under GPL-3.0
"""

__version__ = "1.0.0"

# Import main user-facing API
from .lm import lm, LinearModel

# Import backend utilities (for advanced users)
from ._backends import get_backend, list_available_backends

__all__ = [
    'lm',
    'LinearModel',
    'get_backend',
    'list_available_backends',
]