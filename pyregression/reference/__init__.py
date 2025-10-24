"""
Reference implementation (NumPy) with R compatibility.

Numerically equivalent to R within machine precision.
"""

from .linear_model import LinearModel
from .glm import GLM
from .families import Family, Gaussian, Binomial, Poisson

__all__ = [
    "LinearModel",
    "GLM", 
    "Family",
    "Gaussian",
    "Binomial",
    "Poisson",
]
