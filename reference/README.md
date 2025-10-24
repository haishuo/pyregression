# R Source Code Reference

This directory contains the original R source code that PyRegression is based on.

## Files

All files are from **R version 4.4.2** (released 2024).

### Fortran LINPACK Routines (QR Decomposition)
- `dqrdc.f` - Original LINPACK QR decomposition
- `dqrdc2.f` - Modified version used by R (with column pivoting and rank determination)
- `dqrls.f` - QR least squares solver (calls dqrdc2 and dqrsl)
- `dqrsl.f` - QR solve utilities (applies Q, solves R system)
- `dqrutl.f` - QR utility functions (wrappers for different operations)

### C Implementation
- `lm.c` - C wrapper that calls Fortran routines from R
- `family.c` - GLM family functions (link functions, variance functions, etc.)

### R Interface  
- `lm.R` - High-level R interface for linear models
- `glm.R` - High-level R interface for generalized linear models

## Copyright

These files are from R, which is distributed under GPL-2 or GPL-3.
We include them for reference and documentation purposes only.

PyRegression implements the same algorithms but is an independent implementation.

## Algorithm Summary

**Linear Models (`lm`):**
- QR decomposition via Householder transformations
- Column pivoting for rank determination
- Tolerance: 1e-7 (default)

**GLMs (`glm`):**
- Iteratively Reweighted Least Squares (IRLS)
- Convergence: |dev - dev_old| / (0.1 + |dev|) < 1e-8
- Max iterations: 25 (default)

See `../docs/r_algorithm_analysis.md` for detailed specification.
