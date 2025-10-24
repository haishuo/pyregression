"""
PyRegression: GPU-accelerated statistical inference with R-compatible numerics.

Copyright (C) 2024 SGCX
Licensed under GPL-3.0
"""

__version__ = "0.1.0"

from .lm import LinearModel
from .glm import GLM
from ._backends import get_backend, list_available_backends

__all__ = [
    "LinearModel",
    "GLM",
    "get_backend",
    "list_available_backends",
    "__version__",
]
