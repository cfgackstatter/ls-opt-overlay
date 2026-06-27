# risk.py
import numpy as np

def estimate_covariance_ewma(returns: np.ndarray, halflife: int = 36) -> np.ndarray:
    """EWMA covariance matrix. halflife in periods."""
    if returns.shape[0] < 12:
        raise ValueError(f"Need at least 12 periods, got {returns.shape[0]}")
    
    decay = 0.5 ** (1 / halflife)
    n = returns.shape[0]
    weights = decay ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()

    centered = returns - (weights @ returns)  # weighted mean subtracted
    weighted = centered * weights[:, np.newaxis]  # scale rows by weights
    return weighted.T @ centered  # (n_assets, n_assets)

def estimate_covariance_sample(returns: np.ndarray) -> np.ndarray:
    return np.cov(returns, rowvar=False)