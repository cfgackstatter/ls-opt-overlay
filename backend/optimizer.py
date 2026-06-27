# optimizer.py
import numpy as np
import cvxpy as cp

def optimize_portfolio(
    alphas: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 0.5,
    long_weight: float = 1.0,
    short_weight: float = 1.0,
    max_weight: float = 0.1,
    current_weights: np.ndarray | None = None,
    turnover_limit: float | None = None,
) -> dict:
    """Mean-variance optimization. Pass current_weights + turnover_limit to enable turnover constraint."""
    n = len(alphas)
    w_long = cp.Variable(n, nonneg=True)
    w_short = cp.Variable(n, nonneg=True)
    w = w_long - w_short

    objective = cp.Maximize(alphas @ w - risk_aversion * cp.quad_form(w, cov_matrix))
    constraints = [
        cp.sum(w_long) == long_weight,
        cp.sum(w_short) == short_weight,
        w >= -max_weight,
        w <= max_weight,
    ]
    if current_weights is not None and turnover_limit is not None:
        constraints.append(cp.sum(cp.abs(w - current_weights)) <= turnover_limit)

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except Exception as e:
        print(f"Optimization error: {e}")

    fallback = current_weights if current_weights is not None else np.zeros(n)
    if w.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        print(f"Warning: Optimization status: {problem.status}")
        return {"weights": fallback, "turnover": 0.0, "status": problem.status}

    optimal = w.value
    return {
        "weights": optimal,
        "turnover": float(np.sum(np.abs(optimal - fallback))),
        "expected_return": float(alphas @ optimal),
        "risk": float(optimal @ cov_matrix @ optimal),
        "status": problem.status,
    }