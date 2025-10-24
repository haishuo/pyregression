# PyRegression

GPU-accelerated statistical inference for regression models with numerical validation against R.

## Overview

PyRegression provides:
- **Linear models** (`lm`) - Ordinary least squares regression
- **Generalized linear models** (`glm`) - Logistic, Poisson, Gamma, etc.
- **Reference implementation** (NumPy) - Bit-for-bit compatible with R within machine precision
- **GPU implementation** (PyTorch) - 100-1000x faster, statistically equivalent

## Philosophy

**Validation first, optimization second.**

PyRegression exists to bring rigorous statistical inference to GPU scale. Every function is numerically 
validated against R to establish trust, then optimized for GPU performance while maintaining statistical 
correctness.

## Installation
```bash
pip install pyregression
```

## Quick Start
```python
import numpy as np
from pyregression import LinearModel

# Your data
X = np.random.randn(1000, 10)
y = X @ np.random.randn(10) + np.random.randn(1000)

# Fit model
model = LinearModel()
result = model.fit(X, y)

# Results (same as R)
print(result.coef)        # Coefficients
print(result.se)          # Standard errors  
print(result.pvalues)     # P-values
print(result.r_squared)   # R²
```

## Status

🚧 **In Development** 🚧

Currently implementing reference (NumPy) version with validation against R.

- [ ] Linear models (`lm`)
- [ ] Logistic regression (`glm` with binomial family)
- [ ] Other GLM families
- [ ] GPU (PyTorch) implementation
- [ ] Comprehensive test suite

## Project Structure
```
pyregression/
├── reference/r-source/    # Original R source code (for reference)
├── docs/                  # Algorithm documentation
├── pyregression/
│   ├── reference/         # NumPy implementation (R-compatible)
│   └── gpu/               # PyTorch implementation (GPU-accelerated)
└── tests/                 # Validation test suite
```

## License

GPL-3.0 (to match R)

## Reference

R source code (for algorithm reference) is included in `reference/r-source/`.
Algorithm analysis and specification: see `docs/r_algorithm_analysis.md`.


## Architecture

PyRegression follows the same proven architecture as PyMVNMLE:

**Two-Track System:**
- **CPU Track**: NumPy + LAPACK, always FP64, R-compatible (reference implementation)
- **GPU Track**: PyTorch, FP32/FP64, statistically equivalent (performance)

**Precision-Based Design:**
- FP32 for consumer GPUs (RTX, Metal)
- FP64 for data center GPUs (A100, H100) and CPU
- Automatic selection based on hardware capabilities
```python
from pyregression import LinearModel

# Auto-selects optimal backend
model = LinearModel()  # CPU FP64 by default

# Explicit selection
model = LinearModel(backend='gpu', use_fp64=False)  # GPU FP32
model = LinearModel(backend='cpu')  # CPU FP64 (reference)
```
