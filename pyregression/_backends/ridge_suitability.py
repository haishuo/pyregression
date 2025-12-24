"""
Ridge regression suitability checker.

Determines if ridge regression can safely substitute QR decomposition
for OLS estimation on platforms where QR is unavailable (e.g., Apple Metal).
"""

import numpy as np
from typing import Optional, Dict, List
import warnings


def check_ridge_suitability(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Check if ridge regression can safely substitute QR for this problem.
    
    Evaluates:
    - Condition number (multicollinearity)
    - Feature scaling
    - Variance Inflation Factors (VIF)
    
    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix (WITHOUT intercept)
    y : ndarray, shape (n,)
        Response vector
    weights : ndarray, optional
        Observation weights
    
    Returns
    -------
    dict with keys:
        - suitable: bool - Can ridge safely substitute QR?
        - reason: str - Explanation if not suitable
        - condition_number: float
        - scale_ratio: float
        - max_vif: float
        - recommended_lambda: float
        - warnings: list[str]
    """
    n, p = X.shape
    
    # Add intercept for analysis
    X_full = np.column_stack([np.ones(n), X])
    p_full = X_full.shape[1]
    
    # Apply weights if provided
    if weights is not None:
        good = weights > 0
        if not np.any(good):
            return {
                'suitable': False,
                'reason': "All weights are zero",
                'condition_number': np.inf
            }
        w_sqrt = np.sqrt(weights[good])
        X_work = X_full[good, :] * w_sqrt[:, np.newaxis]
    else:
        X_work = X_full
    
    # For very large n, subsample for condition number estimation
    if X_work.shape[0] > 10000:
        subset_idx = np.random.choice(X_work.shape[0], 10000, replace=False)
        X_sample = X_work[subset_idx]
    else:
        X_sample = X_work
    
    # Form Gram matrix
    XtX = X_sample.T @ X_sample
    
    # 1. CONDITION NUMBER
    try:
        eigvals = np.linalg.eigvalsh(XtX)
        eigvals = eigvals[eigvals > 0]  # Remove numerical zeros
        
        if len(eigvals) == 0:
            cond = np.inf
        else:
            cond = eigvals.max() / eigvals.min()
    except np.linalg.LinAlgError:
        cond = np.inf
    
    # 2. SCALING RATIO
    col_norms = np.linalg.norm(X_work, axis=0)
    col_norms = col_norms[col_norms > 0]  # Exclude zero columns
    
    if len(col_norms) < p_full:
        scale_ratio = np.inf
    else:
        scale_ratio = col_norms.max() / col_norms.min()
    
    # 3. VARIANCE INFLATION FACTORS (VIF)
    try:
        XtX_inv = np.linalg.inv(XtX)
        vif_values = np.diag(XtX_inv)
        max_vif = vif_values.max()
    except np.linalg.LinAlgError:
        max_vif = np.inf
    
    # Collect warnings
    warning_messages: List[str] = []
    
    # DECISION CRITERIA
    
    # Critical: Severe multicollinearity
    if cond > 1e10:
        return {
            'suitable': False,
            'reason': (
                f"Severe multicollinearity detected (κ = {cond:.2e}).\n"
                f"Ridge regularization would significantly alter coefficient estimates.\n"
                f"This problem requires rank-revealing QR decomposition."
            ),
            'condition_number': cond,
            'scale_ratio': scale_ratio,
            'max_vif': max_vif,
            'warnings': []
        }
    
    # Critical: Extreme scaling issues
    if scale_ratio > 1e8:
        return {
            'suitable': False,
            'reason': (
                f"Extreme scaling differences (ratio = {scale_ratio:.2e}).\n"
                f"Features vary by more than 8 orders of magnitude.\n"
                f"Standardize features or use CPU backend with QR."
            ),
            'condition_number': cond,
            'scale_ratio': scale_ratio,
            'max_vif': max_vif,
            'warnings': []
        }
    
    # Warnings: Moderate issues
    if scale_ratio > 1e6:
        warning_messages.append(
            f"Poor feature scaling detected (ratio: {scale_ratio:.2e}). "
            f"Consider standardizing features for better numerical stability."
        )
    
    if max_vif > 100:
        warning_messages.append(
            f"High multicollinearity detected (max VIF: {max_vif:.1f}). "
            f"Ridge penalty λ will be increased for numerical stability."
        )
    
    if cond > 1000:
        warning_messages.append(
            f"Moderate conditioning issues (κ: {cond:.2e}). "
            f"Small ridge penalty will be added for stability."
        )
    
    # DETERMINE LAMBDA
    # Scale by trace to make it scale-invariant
    trace_XtX = np.trace(XtX)
    
    if cond < 100:
        # Well-conditioned: minimal ridge (essentially zero)
        lam = 1e-10 * trace_XtX / p_full
    elif cond < 1000:
        # Moderate: light ridge for stability
        lam = 1e-8 * trace_XtX / p_full
    elif cond < 10000:
        # Poor conditioning: stronger ridge
        lam = 1e-6 * trace_XtX / p_full
    else:
        # Very poor (but still acceptable): even stronger
        lam = 1e-5 * trace_XtX / p_full
    
    return {
        'suitable': True,
        'condition_number': cond,
        'scale_ratio': scale_ratio,
        'max_vif': max_vif,
        'recommended_lambda': float(lam),
        'warnings': warning_messages
    }


def format_suitability_message(suitability: Dict) -> str:
    """
    Format suitability check results as user-friendly message.
    
    Parameters
    ----------
    suitability : dict
        Output from check_ridge_suitability()
    
    Returns
    -------
    str
        Formatted message for user
    """
    if not suitability['suitable']:
        return (
            f"Ridge regression cannot safely substitute QR on this platform.\n\n"
            f"{suitability['reason']}\n\n"
            f"Options:\n"
            f"  1. Use backend='cpu' for exact QR-based OLS\n"
            f"  2. Preprocess data (standardize, remove collinear features)\n"
            f"  3. Use NVIDIA GPU for full QR support"
        )
    else:
        msg = (
            f"Using ridge regression (λ = {suitability['recommended_lambda']:.2e}) "
            f"as substitute for QR decomposition.\n"
            f"Condition number: {suitability['condition_number']:.2e}\n"
        )
        
        if suitability['warnings']:
            msg += "\nWarnings:\n"
            for warning in suitability['warnings']:
                msg += f"  - {warning}\n"
        
        msg += "\nFor exact QR-based OLS, use backend='cpu'."
        
        return msg