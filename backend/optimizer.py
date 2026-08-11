# optimizer.py
"""Mean-variance portfolio optimizer with transaction costs in the objective.

Industry-standard single-period construction (Grinold–Kahn / Boyd et al.):

    max_w  α'w − λ w'Σw − κ₁‖Δw‖₁ − κ₂‖Δw‖₂²

subject to sleeve and name limits. Linear κ₁ is the proportional spread/commission
model (bps of NAV per unit of Σ|Δw|). Optional κ₂ captures simple market impact.
A hard turnover cap is optional and off by default — funds usually let costs
discipline turnover rather than a knife-edge constraint.

κ₁/κ₂ and the hard cap are fixed at construction (constants) so the problem stays
DPP; only α, risk factor L, and w_prev are Parameters.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

_COV_RIDGE = 1e-8
_OSQP_KW = dict(
    solver=cp.OSQP,
    warm_start=True,
    verbose=False,
    eps_abs=1e-5,
    eps_rel=1e-5,
    polishing=False,
    max_iter=8_000,
)


def _chol_factor(cov: np.ndarray, risk_aversion: float) -> np.ndarray:
    """Upper-triangular L s.t. ||L w||^2 = w'(λ Σ)w."""
    sigma = 0.5 * (cov + cov.T) + _COV_RIDGE * np.eye(cov.shape[0])
    scaled = max(float(risk_aversion), 1e-12) * sigma
    try:
        return np.linalg.cholesky(scaled).T
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(scaled)
        eigvals = np.clip(eigvals, _COV_RIDGE, None)
        return (eigvecs * np.sqrt(eigvals)).T


class PortfolioOptimizer:
    """Reusable Markowitz + TC solver for fixed sleeve / cost / bound parameters."""

    def __init__(
        self,
        n_assets: int,
        risk_aversion: float,
        long_weight: float,
        short_weight: float,
        max_long_weight: float,
        max_short_weight: float,
        hard_turnover_limit: float = 0.0,
        tc_linear: float = 0.0,
        tc_quad: float = 0.0,
    ):
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        long_weight = float(max(long_weight, 0.0))
        short_weight = float(max(short_weight, 0.0))
        max_long_weight = float(max(max_long_weight, 0.0))
        max_short_weight = float(max(max_short_weight, 0.0))
        hard_turnover_limit = float(max(hard_turnover_limit, 0.0))
        tc_linear = float(max(tc_linear, 0.0))
        tc_quad = float(max(tc_quad, 0.0))

        if max_long_weight > 0 and long_weight > n_assets * max_long_weight + 1e-9:
            raise ValueError(
                f"long_weight={long_weight} infeasible with "
                f"max_long_weight={max_long_weight} and n={n_assets}"
            )
        if max_short_weight > 0 and short_weight > n_assets * max_short_weight + 1e-9:
            raise ValueError(
                f"short_weight={short_weight} infeasible with "
                f"max_short_weight={max_short_weight} and n={n_assets}"
            )

        self.n = n_assets
        self.risk_aversion = float(risk_aversion)
        self.long_weight = long_weight
        self.short_weight = short_weight
        self.hard_turnover_limit = hard_turnover_limit
        self.tc_linear = tc_linear
        self.tc_quad = tc_quad

        self._alphas = cp.Parameter(n_assets)
        self._L = cp.Parameter((n_assets, n_assets))
        self._w_prev = cp.Parameter(n_assets)

        w_long = cp.Variable(n_assets, nonneg=True)
        w_short = cp.Variable(n_assets, nonneg=True)
        self._w = w_long - w_short
        dw = self._w - self._w_prev

        # Constants (not Parameters) so κ·‖Δw‖ stays DPP
        objective = cp.Maximize(
            self._alphas @ self._w
            - cp.sum_squares(self._L @ self._w)
            - tc_linear * cp.sum(cp.abs(dw))
            - tc_quad * cp.sum_squares(dw)
        )
        constraints = [
            cp.sum(w_long) == self.long_weight,
            cp.sum(w_short) == self.short_weight,
            w_long <= max_long_weight,
            w_short <= max_short_weight,
        ]
        if hard_turnover_limit > 0:
            constraints.append(cp.sum(cp.abs(dw)) <= hard_turnover_limit)

        self._problem = cp.Problem(objective, constraints)
        if not self._problem.is_dcp(dpp=True):
            raise RuntimeError("Portfolio problem must be DPP for fast re-solves")

    def solve(
        self,
        alphas: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: np.ndarray | None = None,
    ) -> dict:
        """Solve one rebalance. Costs κ₁/κ₂ are fixed at construction."""
        prev = (
            np.zeros(self.n)
            if current_weights is None
            else np.asarray(current_weights, dtype=float).copy()
        )
        self._alphas.value = np.asarray(alphas, dtype=float)
        self._L.value = _chol_factor(cov_matrix, self.risk_aversion)
        self._w_prev.value = prev

        try:
            self._problem.solve(**_OSQP_KW)
        except Exception as e:
            print(f"Optimization error: {e}")
            return {"weights": prev, "turnover": 0.0, "tc_penalty": 0.0, "status": "error"}

        if self._w.value is None or self._problem.status not in ("optimal", "optimal_inaccurate"):
            return {
                "weights": prev,
                "turnover": 0.0,
                "tc_penalty": 0.0,
                "status": self._problem.status,
            }

        optimal = np.asarray(self._w.value, dtype=float).ravel()
        turnover = float(np.sum(np.abs(optimal - prev)))
        tc_penalty = (
            self.tc_linear * turnover
            + self.tc_quad * float(np.sum((optimal - prev) ** 2))
        )
        return {
            "weights": optimal,
            "turnover": turnover,
            "tc_penalty": tc_penalty,
            "expected_return": float(alphas @ optimal),
            "risk": float(optimal @ cov_matrix @ optimal),
            "status": self._problem.status,
        }


def optimize_portfolio(
    alphas: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 0.5,
    long_weight: float = 1.0,
    short_weight: float = 0.0,
    max_long_weight: float = 0.1,
    max_short_weight: float | None = None,
    current_weights: np.ndarray | None = None,
    tc_linear: float = 0.0,
    tc_quad: float = 0.0,
    hard_turnover_limit: float = 0.0,
    optimizer: PortfolioOptimizer | None = None,
) -> dict:
    """Mean-variance + TC optimization. Pass a PortfolioOptimizer to reuse the problem."""
    if max_short_weight is None:
        max_short_weight = max_long_weight
    if optimizer is None:
        optimizer = PortfolioOptimizer(
            len(alphas),
            risk_aversion,
            long_weight,
            short_weight,
            max_long_weight,
            max_short_weight,
            hard_turnover_limit,
            tc_linear=tc_linear,
            tc_quad=tc_quad,
        )
    return optimizer.solve(
        alphas,
        cov_matrix,
        current_weights=current_weights,
    )
