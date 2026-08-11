# alpha.py
"""Alpha construction from a cross-sectional signal.

Default is Grinold's formulation (industry / Barra-style):
    α_i = IC · σ_i · z_i
where z is a cross-sectional z-score (or rank-z) and σ_i = √Σ_ii.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

_EPS = 1e-12


def cross_sectional_zscore(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=float).ravel()
    return (x - x.mean()) / (x.std() + _EPS)


def rank_zscore(scores: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(scores, dtype=float).ravel()).astype(float)
    return (ranks - ranks.mean()) / (ranks.std() + _EPS)


def calculate_alphas(
    factor_scores: np.ndarray,
    cov_matrix: np.ndarray,
    *,
    method: str = "grinold",
    signal_ic: float = 0.05,
) -> np.ndarray:
    """
    Map a raw signal to expected excess returns (same units as returns in Σ).

    Methods
    -------
    grinold:
        α = IC · σ · z(score)     [Grinold]
    rank_grinold:
        α = IC · σ · z(rank(score))
    zscore:
        α = z(score)              (scale-free; risk aversion absorbs scale)
    """
    method = (method or "grinold").lower()
    vol = np.sqrt(np.maximum(np.diag(cov_matrix), _EPS))

    if method == "grinold":
        z = cross_sectional_zscore(factor_scores)
        return float(signal_ic) * vol * z
    if method == "rank_grinold":
        z = rank_zscore(factor_scores)
        return float(signal_ic) * vol * z
    if method == "zscore":
        return cross_sectional_zscore(factor_scores)

    raise ValueError(f"Unknown alpha method: {method!r}")
