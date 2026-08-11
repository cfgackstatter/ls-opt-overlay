# risk.py
"""Covariance estimators for portfolio construction."""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def estimate_covariance_ewma(returns: np.ndarray, halflife: int = 36) -> np.ndarray:
    """EWMA covariance (RiskMetrics-style). halflife in periods."""
    if returns.shape[0] < 12:
        raise ValueError(f"Need at least 12 periods, got {returns.shape[0]}")

    decay = 0.5 ** (1.0 / max(halflife, 1))
    n = returns.shape[0]
    weights = decay ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()

    centered = returns - (weights @ returns)
    weighted = centered * weights[:, np.newaxis]
    return weighted.T @ centered


def estimate_covariance_ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit–Wolf shrinkage toward a scaled identity (constant correlation-free target).

    Shrinks the sample covariance to μ·I with optimal intensity — standard
    when N is large relative to T.
    """
    if returns.shape[0] < 12:
        raise ValueError(f"Need at least 12 periods, got {returns.shape[0]}")

    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    x = x - x.mean(axis=0, keepdims=True)
    sample = (x.T @ x) / t

    mu = np.trace(sample) / n
    target = mu * np.eye(n)

    # Frobenius intensities (Ledoit–Wolf 2004, simplified)
    x2 = x**2
    pi_hat = np.sum((x.T @ x / t - sample) ** 2)  # rough; use elementwise var of products
    # Standard LW formula:
    pi_mat = np.zeros((n, n))
    for i in range(t):
        m = np.outer(x[i], x[i]) - sample
        pi_mat += m**2
    pi_hat = pi_mat.sum() / t

    rho_hat = 0.0  # identity target ⇒ no off-diag structure in prior
    # For identity target, rho term for diagonals:
    rho_hat = np.sum(np.diag(pi_mat) / t)

    gamma_hat = np.linalg.norm(sample - target, "fro") ** 2
    kappa = (pi_hat - rho_hat) / (gamma_hat + _EPS)
    shrinkage = float(np.clip(kappa / t, 0.0, 1.0))
    return shrinkage * target + (1.0 - shrinkage) * sample


def estimate_covariance(
    returns: np.ndarray,
    *,
    method: str = "ewma",
    halflife: int = 36,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Dispatch covariance estimator and add a small diagonal ridge."""
    method = (method or "ewma").lower()
    if method == "ewma":
        cov = estimate_covariance_ewma(returns, halflife=halflife)
    elif method in ("ledoit", "ledoit_wolf", "lw"):
        cov = estimate_covariance_ledoit_wolf(returns)
    else:
        raise ValueError(f"Unknown covariance method: {method!r}")

    cov = 0.5 * (cov + cov.T)
    return cov + float(ridge) * np.eye(cov.shape[0])
