"""
Core algorithms (backend-agnostic).
"""

from .qr import qr_decomposition_with_pivoting
from .lm_solver import fit_linear_model

__all__ = [
    "qr_decomposition_with_pivoting",
    "fit_linear_model",
]
