import numpy as np
import cvxpy as cp

def optimize_portfolio(alphas: np.ndarray, 
                      cov_matrix: np.ndarray,
                      current_weights: np.ndarray,
                      risk_aversion: float = 0.5,
                      long_weight: float = 1.0,
                      short_weight: float = 1.0,
                      turnover_limit: float = 0.5,
                      max_weight: float = 0.1) -> dict:
    """
    Mean-variance portfolio optimization with turnover constraint using cvxpy
    
    Uses split variables: w = w_long - w_short where w_long, w_short >= 0
    
    Args:
        alphas: (n_assets,) expected returns
        cov_matrix: (n_assets, n_assets) covariance matrix
        current_weights: (n_assets,) current portfolio weights
        risk_aversion: Risk aversion parameter (default 0.5)
        long_weight: Target long exposure (as decimal, e.g., 1.0 = 100%)
        short_weight: Target short exposure (as decimal, e.g., 1.0 = 100%)
        turnover_limit: Maximum turnover allowed (as decimal)
        max_weight: Maximum absolute weight per asset (as decimal)
    
    Returns:
        Dictionary with 'weights', 'turnover', 'expected_return', 'risk'
    """
    n_assets = len(alphas)
    
    # Split variables: w = w_long - w_short
    w_long = cp.Variable(n_assets, nonneg=True)
    w_short = cp.Variable(n_assets, nonneg=True)
    w = w_long - w_short
    
    portfolio_return = alphas @ w
    portfolio_risk = cp.quad_form(w, cov_matrix)
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
    
    constraints = []
    
    # Long and short exposure constraints
    constraints.append(cp.sum(w_long) == long_weight)
    constraints.append(cp.sum(w_short) == short_weight)
    
    # Turnover constraint
    turnover = cp.sum(cp.abs(w - current_weights))
    constraints.append(turnover <= turnover_limit)
    
    # Position size limits (on net position)
    constraints.append(w >= -max_weight)
    constraints.append(w <= max_weight)
    
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Optimization status: {problem.status}")
            return {
                'weights': current_weights,
                'turnover': 0.0,
                'expected_return': float(alphas @ current_weights),
                'risk': float(current_weights @ cov_matrix @ current_weights),
                'status': problem.status
            }
        
        optimal_weights = w.value
        realized_turnover = np.sum(np.abs(optimal_weights - current_weights))
        
        return {
            'weights': optimal_weights,
            'turnover': float(realized_turnover),
            'expected_return': float(alphas @ optimal_weights),
            'risk': float(optimal_weights @ cov_matrix @ optimal_weights),
            'status': problem.status
        }
    
    except Exception as e:
        print(f"Optimization error: {e}")
        return {
            'weights': current_weights,
            'turnover': 0.0,
            'expected_return': float(alphas @ current_weights),
            'risk': float(current_weights @ cov_matrix @ current_weights),
            'status': 'error'
        }


def optimize_portfolio_simple(alphas: np.ndarray, 
                              cov_matrix: np.ndarray,
                              risk_aversion: float = 0.5,
                              long_weight: float = 1.0,
                              short_weight: float = 1.0,
                              max_weight: float = 0.1) -> np.ndarray:
    """
    Simplified version without turnover constraint (for initial portfolio)
    
    Args:
        alphas: (n_assets,) expected returns
        cov_matrix: (n_assets, n_assets) covariance matrix
        risk_aversion: Risk aversion parameter
        long_weight: Target long exposure (as decimal)
        short_weight: Target short exposure (as decimal)
        max_weight: Maximum absolute weight per asset (as decimal)
    
    Returns:
        (n_assets,) optimal weights
    """
    n_assets = len(alphas)
    
    # Split variables: w = w_long - w_short
    w_long = cp.Variable(n_assets, nonneg=True)
    w_short = cp.Variable(n_assets, nonneg=True)
    w = w_long - w_short
    
    portfolio_return = alphas @ w
    portfolio_risk = cp.quad_form(w, cov_matrix)
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
    
    constraints = [
        cp.sum(w_long) == long_weight,
        cp.sum(w_short) == short_weight,
        w >= -max_weight,
        w <= max_weight
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=False)
    
    if w.value is not None and problem.status in ["optimal", "optimal_inaccurate"]:
        return w.value
    else:
        return np.zeros(n_assets)
