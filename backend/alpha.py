# alpha.py
import numpy as np
from scipy.stats import rankdata

def calculate_alphas(factor_scores: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    ranks = rankdata(factor_scores).astype(float)
    ranks -= ranks.mean()
    ranks /= ranks.std() + 1e-8
    
    port_vol = np.sqrt(abs(ranks @ cov_matrix @ ranks) + 1e-8)
    risk_adjusted = ranks / port_vol
    return cov_matrix @ risk_adjusted

def calculate_alphas_batch(
    factor_scores: np.ndarray,
    returns_history: np.ndarray,
    lookback: int = 36,
    method: str = "ewma",
) -> np.ndarray:
    from backend.risk import estimate_covariance_ewma, estimate_covariance_sample
    
    cov_fn = estimate_covariance_ewma if method == "ewma" else estimate_covariance_sample
    n_periods = factor_scores.shape[0]
    
    if n_periods < lookback:
        raise ValueError(f"Need at least {lookback} periods, got {n_periods}")
    
    return np.array([
        calculate_alphas(factor_scores[t], cov_fn(returns_history[t - lookback:t]))
        for t in range(lookback, n_periods)
    ])