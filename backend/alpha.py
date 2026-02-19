import numpy as np

def calculate_alphas(factor_scores: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate alphas from factor scores using rank-based approach
    
    Higher factor score → Higher alpha (positive relationship preserved)
    
    Steps:
    1. Rank by factor score (higher score = higher rank)
    2. Demean and standardize ranks
    3. Risk-adjust: normalize by portfolio volatility
    4. Transform to alpha space: alpha = Cov * risk_adjusted_ranks
    
    Args:
        factor_scores: (n_assets,) current factor scores
        cov_matrix: (n_assets, n_assets) covariance matrix
    
    Returns:
        (n_assets,) alpha vector (higher factor → higher alpha)
    """
    n_assets = len(factor_scores)
    
    # Step 1: Rank (higher factor = higher rank)
    # argsort gives indices that would sort the array
    # argsort twice gives ranks
    ranks = np.argsort(np.argsort(factor_scores)).astype(float)
    # Now ranks[i] is the rank of asset i (0 = lowest, n_assets-1 = highest)
    
    # Step 2: Demean and standardize
    ranks_centered = ranks - ranks.mean()
    ranks_std = ranks_centered / (ranks_centered.std() + 1e-8)
    
    # Step 3: Risk adjustment
    # Normalize by the portfolio volatility if we held these weights
    portfolio_variance = ranks_std @ cov_matrix @ ranks_std
    portfolio_vol = np.sqrt(np.abs(portfolio_variance) + 1e-8)
    risk_adjusted = ranks_std / portfolio_vol
    
    # Step 4: Transform to alpha space
    # This gives expected returns in the same units as the covariance matrix
    alphas = cov_matrix @ risk_adjusted
    
    return alphas


def calculate_alphas_batch(factor_scores: np.ndarray, returns_history: np.ndarray, 
                          lookback: int = 36, method: str = 'ewma') -> np.ndarray:
    """
    Calculate alphas for multiple time periods
    
    Args:
        factor_scores: (n_periods, n_assets) factor scores over time
        returns_history: (n_periods, n_assets) returns history (aligned with factors)
        lookback: Number of periods to use for covariance estimation
        method: 'ewma' or 'sample'
    
    Returns:
        (n_periods - lookback, n_assets) alpha matrix
    """
    from backend.risk import estimate_covariance_ewma, estimate_covariance_sample
    
    n_periods, n_assets = factor_scores.shape
    
    if n_periods < lookback:
        raise ValueError(f"Need at least {lookback} periods, got {n_periods}")
    
    alphas = np.zeros((n_periods - lookback, n_assets))
    
    for t in range(lookback, n_periods):
        # Use returns from [t-lookback : t] to estimate covariance
        returns_window = returns_history[t-lookback:t]
        
        if method == 'ewma':
            cov = estimate_covariance_ewma(returns_window)
        else:
            cov = estimate_covariance_sample(returns_window)
        
        # Calculate alpha using factor score at time t
        alphas[t - lookback] = calculate_alphas(factor_scores[t], cov)
    
    return alphas
