"""
PyRegression: GPU-accelerated statistical inference with R-compatible numerics.

Copyright (C) 2024 SGCX
Licensed under GPL-3.0
"""

__version__ = "0.1.0"

# Reference implementation (NumPy, R-compatible)
from pyregression.reference.linear_model import LinearModel
from pyregression.reference.glm import GLM

__all__ = [
    "LinearModel",
    "GLM",
]
