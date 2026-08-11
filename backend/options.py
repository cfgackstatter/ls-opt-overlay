# options.py
"""Equity options overlay: theoretical pricing + sticky-alpha overwrite.

Thesis
------
After the optimizer keeps a name (often because turnover / TC makes a full exit
suboptimal), overwrite positions whose alpha is no longer attractive:

  • Long  + α ≤ call_barrier  → sell OTM covered calls  (cap upside you don't want)
  • Short + α ≥ put_barrier   → sell OTM covered puts   (covered-put dual)

This asks whether harvesting option premium on *sticky, low-conviction* holdings
improves P&L versus the equity book alone.

Pricing (industry theoretical backtest)
---------------------------------------
European Black–Scholes–Merton mid on each name:

    C/P = BSM(S, K, T, r, q, σ̂)

  • T = 1/12 (written at month-start, cash-settled at next rebalance)
  • σ̂ = annualized EWMA/Ledoit asset vol from the equity risk model
  • r = financing cash rate; q = dividend yield (0 matches this project's
    non-dividend GBM — keep q=0 unless the DGP pays dividends)
  • Trade at the bid: mid × (1 − ½·spread_bps/1e4), minus per-contract fees
  • Expiry: European cash settlement at exact intrinsic (no bid–ask). Optional
    exercise fee only when ITM.

Under RN pricing, unconditional short options have ~zero excess return before
costs; any edge here must come from *conditioning on alpha* (physical drift
≠ r) plus cost drag.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

SHARES_PER_CONTRACT = 100
TENOR_YEARS = 1.0 / 12.0


def bsm_price(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: float,
    r: float,
    sigma: np.ndarray | float,
    is_call: bool,
    q: float = 0.0,
) -> np.ndarray:
    """Vectorized Black–Scholes–Merton (continuous dividend yield q)."""
    S = np.atleast_1d(np.asarray(S, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=float))
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    shape = np.broadcast(S, K, sigma).shape
    S = np.broadcast_to(S, shape).copy()
    K = np.broadcast_to(K, shape).copy()
    sigma = np.broadcast_to(sigma, shape).copy()

    intrinsic = np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)
    price = np.array(intrinsic, dtype=float, copy=True)
    valid = (S > 0) & (K > 0) & (sigma > 0) & (T > 0)
    if not np.any(valid):
        return price

    Sv, Kv, sigv = S[valid], K[valid], sigma[valid]
    sqrt_t = np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(Sv / Kv) + (r - q + 0.5 * sigv**2) * T) / (sigv * sqrt_t)
    d2 = d1 - sigv * sqrt_t
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    if is_call:
        price[valid] = df_q * Sv * norm.cdf(d1) - df_r * Kv * norm.cdf(d2)
    else:
        price[valid] = df_r * Kv * norm.cdf(-d2) - df_q * Sv * norm.cdf(-d1)
    return np.maximum(price, 0.0)


def sell_options_overlay(
    shares: np.ndarray,
    prices: np.ndarray,
    alphas: np.ndarray,
    cov_matrix: np.ndarray,
    call_otm_pct: float,
    put_otm_pct: float,
    call_alpha_barrier: float,
    put_alpha_barrier: float,
    risk_free_rate: float,
    contract_fee: float,
    spread_bps: float,
    dividend_yield: float = 0.0,
) -> dict:
    """
    Write 1M covered calls/puts on sticky weak-conviction names.

    Coverage is lot-rounded: floor(|shares|/100) contracts per name so the
    equity position always covers the short option.
    """
    shares = np.asarray(shares, dtype=float)
    prices = np.asarray(prices, dtype=float)
    alphas = np.asarray(alphas, dtype=float)

    # Monthly cov → annualized vol for BSM
    asset_vols = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0)) * np.sqrt(12.0)
    n_contracts = (np.abs(shares) // SHARES_PER_CONTRACT).astype(int)

    # Sticky longs with weak/negative alpha → overwrite calls
    call_mask = (shares > 0) & (alphas <= call_alpha_barrier) & (n_contracts > 0)
    # Sticky shorts with strong/positive alpha → covered puts
    put_mask = (shares < 0) & (alphas >= put_alpha_barrier) & (n_contracts > 0)

    option_positions: list[dict] = []
    premium_collected = 0.0
    half_spread = 0.5 * float(spread_bps) / 10_000.0

    for mask, is_call, otm_pct in (
        (call_mask, True, call_otm_pct),
        (put_mask, False, put_otm_pct),
    ):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        S = prices[idxs]
        strike_mult = (1.0 + otm_pct) if is_call else (1.0 - otm_pct)
        K = S * strike_mult
        mid = bsm_price(
            S, K, TENOR_YEARS, risk_free_rate, asset_vols[idxs], is_call,
            q=dividend_yield,
        )
        # Sell at bid relative to theoretical mid
        bid = mid * (1.0 - half_spread)
        nc = n_contracts[idxs]
        net = bid * nc * SHARES_PER_CONTRACT - nc * float(contract_fee)
        premium_collected += float(net.sum())
        option_positions.extend(
            {
                "asset_idx": int(i),
                "type": "call" if is_call else "put",
                "contracts": int(c),
                "strike": float(k),
                "spot_at_sale": float(s),
                "mid_price": float(m),
                "bid_price": float(b),
                "iv": float(v),
            }
            for i, c, k, s, m, b, v in zip(
                idxs, nc, K, S, mid, bid, asset_vols[idxs]
            )
        )

    return {
        "premium_collected": premium_collected,
        "option_positions": option_positions,
        "num_contracts": int(sum(p["contracts"] for p in option_positions)),
    }


def settle_options(
    option_positions: list[dict],
    expiry_prices: np.ndarray,
    contract_fee: float = 0.0,
    exercise_fee: bool = True,
) -> float:
    """
    European cash settlement at intrinsic.

    Returns cash flow to the short option seller (≤ 0). No bid–ask at expiry.
    If exercise_fee, charge contract_fee per ITM contract (clearing-style).
    """
    if not option_positions:
        return 0.0
    expiry_prices = np.asarray(expiry_prices, dtype=float)
    total = 0.0
    fee = float(contract_fee) if exercise_fee else 0.0
    for p in option_positions:
        spot = float(expiry_prices[p["asset_idx"]])
        if p["type"] == "call":
            intrinsic = max(0.0, spot - p["strike"])
        else:
            intrinsic = max(0.0, p["strike"] - spot)
        n = int(p["contracts"])
        total -= intrinsic * n * SHARES_PER_CONTRACT
        if intrinsic > 0 and fee > 0:
            total -= n * fee
    return float(total)
