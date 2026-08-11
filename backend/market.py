# market.py
"""Industry-style synthetic equity market generator.

Multi-factor log-return DGP (monthly):
    log r_{i,t} = μ_i − ½ σ²_{i,t}
                 + β_i' F_t
                 + σ_{i,t} (IC · z_{i,t} + √(1−IC²) · ε_{i,t})

where:
  - F_t = (market, size, value) common factors
  - z_{i,t} = cross-sectional z-score of an AR(1) alpha signal (no look-ahead)
  - ε ~ scaled Student-t (fat tails, unit variance)
  - σ_{i,t} = σ_i · exp(½ v_t) with shared AR(1) log-vol factor v_t
  - prices via GBM: P_{t+1} = P_t · exp(log r_t)

IC is a *residual* information coefficient (signal vs idiosyncratic shock).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MONTHS_PER_YEAR = 12
EPSILON = 1e-12
MIN_STOCK_PRICE = 5.0
MAX_STOCK_PRICE = 200.0
MIN_PRICE_FLOOR = 0.01
ASSET_VOL_SCALING_MIN = 0.5
ASSET_VOL_SCALING_MAX = 1.5


def _ar1_series(
    n_periods: int,
    n_series: int,
    rho: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary Gaussian AR(1): x_t = ρ x_{t-1} + √(1−ρ²)·scale·ε_t."""
    rho = float(np.clip(rho, -0.99, 0.99))
    innov = np.sqrt(max(1.0 - rho**2, 0.0)) * scale
    out = np.empty((n_periods, n_series))
    out[0] = rng.normal(0.0, scale, n_series)
    for t in range(1, n_periods):
        out[t] = rho * out[t - 1] + innov * rng.standard_normal(n_series)
    return out


def generate_ar1_factors(
    n_periods: int,
    n_assets: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit-variance AR(1) alpha signals (one series per asset)."""
    return _ar1_series(n_periods, n_assets, rho, scale=1.0, rng=rng)


def cross_sectional_zscore(scores: np.ndarray) -> np.ndarray:
    """Z-score each row (period) across columns (assets). No look-ahead."""
    mu = scores.mean(axis=1, keepdims=True)
    sd = scores.std(axis=1, keepdims=True)
    return (scores - mu) / (sd + EPSILON)


def _student_t_unit(df: float, size: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Student-t draws scaled to variance 1. Large df → near-Gaussian."""
    df = float(max(df, 2.05))
    raw = rng.standard_t(df, size=size)
    return raw * np.sqrt((df - 2.0) / df)


@dataclass(frozen=True)
class MarketData:
    """Outputs consumed by the strategy / simulator."""

    factor_scores: np.ndarray      # (T, N) alpha signal levels
    log_returns: np.ndarray        # (T, N)
    simple_returns: np.ndarray     # (T, N) = exp(log_r) - 1
    prices: np.ndarray             # (T+1, N)
    betas: np.ndarray              # (N, 3) market / size / value
    common_factors: np.ndarray     # (T, 3)
    idio_vol_annual: np.ndarray    # (N,) unconditional idiosyncratic vol (ann.)


def generate_market_data(
    n_periods: int,
    n_assets: int,
    *,
    mean_return: float = 0.06,
    volatility: float = 0.20,
    ic: float = 0.12,
    factor_autocorr: float = 0.7,
    market_vol: float = 0.15,
    market_autocorr: float = 0.05,
    style_vol: float = 0.07,
    style_autocorr: float = 0.10,
    avg_beta: float = 1.0,
    beta_dispersion: float = 0.30,
    student_t_df: float = 5.0,
    stoch_vol_persistence: float = 0.85,
    stoch_vol_of_vol: float = 0.25,
    rng: np.random.Generator,
) -> MarketData:
    """
    Simulate a multi-factor equity universe.

    `volatility` is the target *total* annualized vol per name (heterogeneous
    ±50%). Idiosyncratic vol is backed out after assigning factor betas so
    total unconditional vol stays near the target.
    """
    if n_periods < 1 or n_assets < 1:
        raise ValueError("n_periods and n_assets must be >= 1")
    ic = float(np.clip(ic, -0.99, 0.99))

    # --- Common factors F_t = (market, size, value), monthly vol ---
    mkt_m = market_vol / np.sqrt(MONTHS_PER_YEAR)
    sty_m = style_vol / np.sqrt(MONTHS_PER_YEAR)
    market = _ar1_series(n_periods, 1, market_autocorr, mkt_m, rng).ravel()
    size = _ar1_series(n_periods, 1, style_autocorr, sty_m, rng).ravel()
    value = _ar1_series(n_periods, 1, style_autocorr, sty_m, rng).ravel()
    common = np.column_stack([market, size, value])  # (T, 3)

    # --- Factor loadings ---
    beta_m = rng.normal(avg_beta, beta_dispersion, n_assets)
    beta_m = np.clip(beta_m, 0.2, 2.0)
    beta_s = rng.normal(0.0, 0.5, n_assets)
    beta_v = rng.normal(0.0, 0.5, n_assets)
    betas = np.column_stack([beta_m, beta_s, beta_v])

    # Target total monthly vol per name; back out idio vol
    target_ann = rng.uniform(
        ASSET_VOL_SCALING_MIN * volatility,
        ASSET_VOL_SCALING_MAX * volatility,
        n_assets,
    )
    target_m = target_ann / np.sqrt(MONTHS_PER_YEAR)
    # Unconditional var of common part (factors ~ unit scaled by their scales already)
    # Each factor series has std ≈ scale (stationary AR(1))
    f_var = np.array([mkt_m**2, sty_m**2, sty_m**2])
    sys_var = (betas**2 * f_var).sum(axis=1)
    idio_var = np.maximum(target_m**2 - sys_var, (0.25 * target_m) ** 2)
    idio_vol = np.sqrt(idio_var)  # monthly
    idio_vol_annual = idio_vol * np.sqrt(MONTHS_PER_YEAR)

    # --- Shared stochastic log-vol factor (one-factor SV) ---
    # v_t AR(1); E[exp(v)] ≈ 1 for small innovations → centers idio vol
    v = _ar1_series(
        n_periods, 1, stoch_vol_persistence, stoch_vol_of_vol, rng
    ).ravel()
    # Center so mean exp(v) ≈ 1
    v = v - np.log(np.mean(np.exp(v)) + EPSILON)
    vol_mult = np.exp(0.5 * v)  # (T,)
    idio_vol_t = idio_vol[np.newaxis, :] * vol_mult[:, np.newaxis]  # (T, N)

    # --- Alpha signal: AR(1), cross-sectionally z-scored each month ---
    factor_scores = generate_ar1_factors(n_periods, n_assets, factor_autocorr, rng)
    z = cross_sectional_zscore(factor_scores)

    # Residual shock with target residual IC
    eps = _student_t_unit(student_t_df, (n_periods, n_assets), rng)
    resid = ic * z + np.sqrt(1.0 - ic**2) * eps

    # Systematic + idiosyncratic log returns with Ito correction
    systematic = common @ betas.T  # (T, N)
    # Conditional variance ≈ sys (known) + idio_vol_t² (SV path)
    cond_var = sys_var[np.newaxis, :] + idio_vol_t**2
    mu_month = mean_return / MONTHS_PER_YEAR
    log_returns = (mu_month - 0.5 * cond_var) + systematic + idio_vol_t * resid

    simple_returns = np.expm1(log_returns)

    # GBM prices
    initial = rng.uniform(MIN_STOCK_PRICE, MAX_STOCK_PRICE, n_assets)
    prices = np.empty((n_periods + 1, n_assets))
    prices[0] = initial
    prices[1:] = initial * np.exp(np.cumsum(log_returns, axis=0))
    prices = np.maximum(prices, MIN_PRICE_FLOOR)

    return MarketData(
        factor_scores=factor_scores,
        log_returns=log_returns,
        simple_returns=simple_returns,
        prices=prices,
        betas=betas,
        common_factors=common,
        idio_vol_annual=idio_vol_annual,
    )


# Backwards-compatible wrappers used by notebooks / older call sites
def generate_correlated_returns(
    factor_scores: np.ndarray,
    ic: float,
    mean: float,
    asset_vols: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Deprecated thin helper: CS-z IC mix → simple returns (no common factors).
    Prefer generate_market_data().
    """
    z = cross_sectional_zscore(factor_scores)
    eps = rng.standard_normal(factor_scores.shape)
    log_r = (mean - 0.5 * asset_vols**2) + asset_vols * (
        ic * z + np.sqrt(max(1.0 - ic**2, 0.0)) * eps
    )
    return np.expm1(log_r)


def generate_prices(returns: np.ndarray, initial_prices: np.ndarray) -> np.ndarray:
    """Build prices from simple returns (legacy). New code uses GBM via log returns."""
    prices = np.empty((len(returns) + 1, len(initial_prices)))
    prices[0] = initial_prices
    prices[1:] = initial_prices * np.cumprod(1.0 + returns, axis=0)
    return np.maximum(prices, MIN_PRICE_FLOOR)
