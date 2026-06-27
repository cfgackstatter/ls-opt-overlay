# options.py
import numpy as np
from scipy.stats import norm

SHARES_PER_CONTRACT = 100
T = 1.0 / 12.0  # 1-month expiry

def _bs_price(S, K, T, r, sigma, is_call: bool) -> np.ndarray:
    """Vectorized Black-Scholes for arrays of S, K, sigma."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return np.where(T > 0, S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), np.maximum(S - K, 0))
    return np.where(T > 0, K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), np.maximum(K - S, 0))

def sell_options_overlay(shares, prices, alphas, cov_matrix,
                         call_otm_pct, put_otm_pct, call_alpha_barrier,
                         put_alpha_barrier, risk_free_rate, contract_fee, spread_bps) -> dict:
    asset_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(12)
    n_contracts = (np.abs(shares) // SHARES_PER_CONTRACT).astype(int)

    call_mask = (shares > 0) & (alphas < call_alpha_barrier) & (n_contracts > 0)
    put_mask  = (shares < 0) & (alphas > put_alpha_barrier)  & (n_contracts > 0)

    option_positions, premium_collected = [], 0.0

    for mask, is_call, otm_pct in [(call_mask, True, call_otm_pct), (put_mask, False, put_otm_pct)]:
        idxs = np.where(mask)[0]
        if not len(idxs):
            continue
        S = prices[idxs]
        strike_mult = (1 + otm_pct) if is_call else (1 - otm_pct)
        K = S * strike_mult
        option_price = _bs_price(S, K, T, risk_free_rate, asset_vols[idxs], is_call)
        nc = n_contracts[idxs]
        gross = option_price * nc * SHARES_PER_CONTRACT
        net = gross * (1 - spread_bps / 10000) - nc * contract_fee
        premium_collected += net.sum()
        option_positions += [
            {"asset_idx": int(i), "type": "call" if is_call else "put",
             "contracts": int(c), "strike": float(k), "spot_at_sale": float(s)}
            for i, c, k, s in zip(idxs, nc, K, S)
        ]

    return {"premium_collected": premium_collected,
            "option_positions": option_positions,
            "num_contracts": sum(p["contracts"] for p in option_positions)}

def settle_options(option_positions: list[dict], expiry_prices: np.ndarray,
                   contract_fee: float, spread_bps: float) -> float:
    if not option_positions:
        return 0.0
    total = 0.0
    for p in option_positions:
        price = expiry_prices[p["asset_idx"]]
        intrinsic = max(0, price - p["strike"]) if p["type"] == "call" else max(0, p["strike"] - price)
        if intrinsic > 0:
            gross = intrinsic * p["contracts"] * SHARES_PER_CONTRACT
            total += gross * (1 + spread_bps / 10000) + p["contracts"] * contract_fee
    return -total