"""
Utility functions.
"""

import numpy as np


def check_array(X, name='X', dtype=np.float64):
    """Validate array input."""
    X = np.asarray(X, dtype=dtype)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or Inf")
    return X


def check_vector(y, name='y', dtype=np.float64):
    """Validate vector input."""
    y = np.asarray(y, dtype=dtype)
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name} contains NaN or Inf")
    return y
