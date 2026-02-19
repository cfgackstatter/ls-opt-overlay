import numpy as np
from numba import jit

def estimate_covariance_ewma(returns: np.ndarray, halflife: int = 36) -> np.ndarray:
    """
    Estimate covariance matrix using exponentially weighted moving average
    
    Args:
        returns: (n_periods, n_assets) array of returns
        halflife: Half-life for exponential weighting in months
    
    Returns:
        (n_assets, n_assets) covariance matrix
    """
    n_periods, n_assets = returns.shape
    
    if n_periods < 12:
        raise ValueError(f"Need at least 12 periods for covariance estimation, got {n_periods}")
    
    # Decay factor
    decay = 0.5 ** (1 / halflife)
    
    # Use Numba-optimized function
    cov = compute_ewma_cov_numba(returns, decay)
    
    return cov


@jit(nopython=True)
def compute_ewma_cov_numba(returns: np.ndarray, decay: float) -> np.ndarray:
    """Numba-optimized EWMA covariance computation"""
    n_periods, n_assets = returns.shape
    
    # Exponential weights (more recent = higher weight)
    weights = np.zeros(n_periods)
    for i in range(n_periods):
        weights[n_periods - 1 - i] = decay ** i
    weights = weights / weights.sum()
    
    # Weighted mean
    weighted_mean = np.zeros(n_assets)
    for t in range(n_periods):
        weighted_mean += weights[t] * returns[t]
    
    # Center returns
    centered = returns - weighted_mean
    
    # Weighted covariance
    cov = np.zeros((n_assets, n_assets))
    for t in range(n_periods):
        for i in range(n_assets):
            for j in range(n_assets):
                cov[i, j] += weights[t] * centered[t, i] * centered[t, j]
    
    return cov


def estimate_covariance_sample(returns: np.ndarray) -> np.ndarray:
    """
    Simple sample covariance matrix
    
    Args:
        returns: (n_periods, n_assets) array of returns
    
    Returns:
        (n_assets, n_assets) covariance matrix
    """
    return np.cov(returns, rowvar=False)
