# R Algorithm Analysis and Specification

Based on analysis of R 4.4.2 source code.

## LINEAR MODELS (`lm`)

### Algorithm: QR Decomposition via Householder Transformations

**Source:** `dqrdc2.f`, `dqrls.f`

### Step 1: QR Decomposition with Column Pivoting

**Method:** Householder transformations (NOT Givens rotations)

**Column pivoting strategy:**
- Compute initial column norms: `||x_j||` for each column j
- At each iteration k:
  - Find column with maximum remaining norm
  - Swap it to position k
  - Apply Householder transformation
  - Update remaining column norms

**Rank determination:**
- Column k contributes to rank if: `current_norm(k) >= original_norm(k) * tol`
- Default tolerance: `tol = 1e-7`

**Numerical stability detail (dqrdc2.f line 147):**
```fortran
if (dabs(t) .lt. 1d-6) go to 130
    qraux(j) = qraux(j)*dsqrt(t)
go to 140
130 continue
    qraux(j) = dnrm2(n-l,x(l+1,j),1)
```
When updating column norms after Householder transformation:
- If relative change < 1e-6: recompute norm from scratch
- Otherwise: use incremental update formula

This prevents accumulation of rounding errors in norm updates.

### Step 2: Solve R β = Q'y

Back-substitution through upper triangular matrix R.

Set coefficients for aliased (rank-deficient) columns to NA.

### Step 3: Compute Statistics

**Residuals:** `r = y - X β`

**Residual Sum of Squares:**
- Unweighted: `RSS = Σ r²`
- Weighted: `RSS = Σ w_i r_i²`

**Residual variance:** `σ² = RSS / df_residual`
where `df_residual = n - rank`

**Coefficient covariance matrix:** `Cov(β) = σ² * (R'R)⁻¹`

**Standard errors:** `SE(β_j) = sqrt(Cov(β)_jj)`

**t-statistics:** `t_j = β_j / SE(β_j)`

**p-values:** `p_j = 2 * P(|T| > |t_j|)` where `T ~ t(df_residual)`

---

## GENERALIZED LINEAR MODELS (`glm`)

### Algorithm: Iteratively Reweighted Least Squares (IRLS)

**Source:** `glm.R` (`glm.fit` function, lines 118-350)

### Initialization

**Starting values for linear predictor η:**
1. If `etastart` provided: use it
2. Else if `start` (coefficients) provided: `η = X β_start + offset`
3. Else if `mustart` provided: `η = link(μ_start)`
4. Else: use family-specific initialization

**For logistic regression:** `μ_start = (y + 0.5) / 2`
(Adds 0.5 to avoid starting at 0 or 1)

### IRLS Loop

Maximum iterations: 25 (default, controlled by `control$maxit`)
```
For iter = 1 to maxiter:
    
    1. Compute working weights and response:
       μ = linkinv(η)
       V = variance(μ)          # Family-specific variance function
       dμ/dη = mu.eta(η)        # Derivative of inverse link
       
       good = (weights > 0) & (dμ/dη != 0) & finite(V) & (V > 0)
       
       w = sqrt(weights * (dμ/dη)² / V)    # Working weights
       z = η + (y - μ) / (dμ/dη)           # Working response
    
    2. Weighted least squares:
       Solve: minimize ||W(z - Xβ)||²
       Using QR decomposition with tolerance = min(1e-7, ε/1000)
       where ε is convergence tolerance (default 1e-8)
    
    3. Update:
       β_new[pivot] = coefficients from QR
       η = X β_new + offset
       μ = linkinv(η)
       dev = Σ deviance_residuals(y_i, μ_i, w_i)
    
    4. Check convergence:
       if |dev - dev_old| / (0.1 + |dev|) < ε:
           converged = TRUE
           BREAK
    
    5. Step-halving if needed:
       if !finite(dev):
           β_new = (β_new + β_old) / 2
           Re-compute η, μ, dev
           Repeat until dev is finite
    
    6. Boundary check:
       if !valideta(η) or !validmu(μ):
           β_new = (β_new + β_old) / 2
           Re-compute η, μ
           Repeat until valid
    
    7. Prepare for next iteration:
       dev_old = dev
       β_old = β_new
```

### Convergence Criterion (Critical Detail)

**From glm.R line 267:**
```r
if (abs(dev - devold)/(0.1 + abs(dev)) < control$epsilon)
```

Default: `control$epsilon = 1e-8`

**Note:** The `0.1 +` term means the relative tolerance is computed with respect to 
`max(0.1, |deviance|)`. This is somewhat arbitrary but we replicate it exactly for 
R compatibility.

### Final Calculations

**Residual types:**

1. **Deviance residuals:**
```
   sign(y - μ) * sqrt(deviance_component(y, μ, w))
```

2. **Pearson residuals:**
```
   (y - μ) * sqrt(w) / sqrt(V(μ))
```

3. **Working residuals:**
```
   z - Xβ  (on scale of linear predictor)
```

4. **Response residuals:**
```
   y - μ
```

**Coefficient covariance matrix:**
```
Cov(β) = dispersion * (R'R)⁻¹
```

Where dispersion:
- Binomial/Poisson: 1 (canonical)
- Others: `Σ w_i r_i² / df_residual`

**Standard errors:** `SE(β) = sqrt(diag(Cov(β)))`

---

## LOGISTIC REGRESSION SPECIFICS

### Link Function: Logit

**From family.c (logit_linkinv, lines 53-65):**
```c
MTHRESH = -30
THRESH = 30
INVEPS = 1/DBL_EPSILON ≈ 4.5e15

if (η < MTHRESH):
    μ = DBL_EPSILON ≈ 2.22e-16
elif (η > THRESH):
    μ = 1 - DBL_EPSILON
else:
    μ = 1 / (1 + exp(-η))
```

**Rationale:** Prevents overflow in `exp(η)` and ensures μ stays in (0, 1).

### Derivative: dμ/dη

**From family.c (logit_mu_eta, lines 67-80):**
```c
if (η > THRESH or η < MTHRESH):
    dμ/dη = DBL_EPSILON
else:
    dμ/dη = exp(η) / (1 + exp(η))²
```

**Alternative formula (numerically stable):**
```
dμ/dη = μ * (1 - μ)
```

### Variance Function

For binomial family: `V(μ) = μ(1 - μ)`

---

## CRITICAL IMPLEMENTATION NOTES

### 1. Machine Epsilon

R uses `DBL_EPSILON ≈ 2.220446e-16`

Python equivalent: `np.finfo(np.float64).eps`

### 2. Tolerance Hierarchy

- **QR tolerance in lm:** `1e-7` (default)
- **QR tolerance in glm IRLS:** `min(1e-7, convergence_tol/1000)` = `1e-11`
- **GLM convergence tolerance:** `1e-8` (default)
- **Norm update threshold (QR):** `1e-6` (triggers full recomputation)

### 3. The Convergence Quirk

The GLM convergence criterion includes `0.1 +` in the denominator:
```python
abs(dev - dev_old) / (0.1 + abs(dev)) < epsilon
```

This is questionable but we replicate it exactly for the reference implementation.

For the GPU implementation, we may offer an improved criterion as an option.

### 4. Numerical Stability in QR

When updating column norms during QR factorization:
- Incremental update can accumulate errors
- If relative change < 1e-6: recompute from scratch
- This matches R's behavior exactly

### 5. Pivoting Strategy

R's `dqrdc2` pivots columns with small norms to the right, effectively:
- Maintaining numerical stability
- Determining rank automatically
- Handling collinearity gracefully

---

## VALIDATION REQUIREMENTS

### Reference Implementation (NumPy)

Must match R within **machine precision** (`≈ 1e-15` for float64):
- Coefficients
- Standard errors
- Residuals
- All statistics (R², AIC, deviance, etc.)

### GPU Implementation (PyTorch)

Must match reference implementation within **statistical tolerance** (`≈ 1e-8`):
- Different algorithms allowed (e.g., Cholesky instead of QR)
- Forward mode allowed (vs R's inverse parameterization)
- Must document any algorithmic differences
- Statistical properties must be preserved (coverage, Type I error, etc.)

---

## REFERENCES

1. **R Source Code:** R-4.4.2, available at https://cran.r-project.org/src/base/
2. **LINPACK:** Dongarra et al. (1979), LINPACK Users' Guide
3. **GLM Theory:** McCullagh & Nelder (1989), Generalized Linear Models, 2nd ed.
4. **QR Decomposition:** Golub & Van Loan (1996), Matrix Computations, 3rd ed.
