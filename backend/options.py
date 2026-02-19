import numpy as np
from scipy.stats import norm

SHARES_PER_CONTRACT = 100


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call option price"""
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put option price"""
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


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
    spread_bps: float
) -> dict:
    """
    Sell options on positions where alpha conditions are met
    
    Returns dict with:
    - premium_collected: Total cash received
    - option_positions: List of sold options for settlement
    - num_contracts: Total contracts sold
    """
    n_assets = len(shares)
    premium_collected = 0.0
    option_positions = []
    num_contracts = 0
    
    # Extract asset-specific volatilities (annualized)
    asset_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(12)
    
    T = 1.0 / 12.0  # 1 month to expiry
    
    for i in range(n_assets):
        if shares[i] == 0:
            continue
        
        spot = prices[i]
        vol = asset_vols[i]
        alpha = alphas[i]
        
        # Long positions: sell calls if alpha < barrier
        if shares[i] > 0 and alpha < call_alpha_barrier:
            # Number of contracts (round down to whole contracts)
            n_contracts = int(abs(shares[i]) / SHARES_PER_CONTRACT)
            
            if n_contracts > 0:
                # Call strike
                strike = spot * (1 + call_otm_pct / 100)
                
                # BS price per share
                call_price_per_share = black_scholes_call(spot, strike, T, risk_free_rate, vol)
                
                # Total premium for all contracts (100 shares each)
                gross_premium = call_price_per_share * n_contracts * SHARES_PER_CONTRACT
                
                # Transaction costs
                total_contract_fees = n_contracts * contract_fee
                spread_cost = gross_premium * (spread_bps / 10000)
                
                net_premium = gross_premium - total_contract_fees - spread_cost
                premium_collected += net_premium
                
                option_positions.append({
                    'asset_idx': i,
                    'type': 'call',
                    'contracts': n_contracts,
                    'strike': strike,
                    'spot_at_sale': spot
                })
                num_contracts += n_contracts
        
        # Short positions: sell puts if alpha > barrier (less negative alpha)
        elif shares[i] < 0 and alpha > put_alpha_barrier:
            n_contracts = int(abs(shares[i]) / SHARES_PER_CONTRACT)
            
            if n_contracts > 0:
                # Put strike
                strike = spot * (1 - put_otm_pct / 100)
                
                # BS price per share
                put_price_per_share = black_scholes_put(spot, strike, T, risk_free_rate, vol)
                
                # Total premium
                gross_premium = put_price_per_share * n_contracts * SHARES_PER_CONTRACT
                
                # Transaction costs
                total_contract_fees = n_contracts * contract_fee
                spread_cost = gross_premium * (spread_bps / 10000)
                
                net_premium = gross_premium - total_contract_fees - spread_cost
                premium_collected += net_premium
                
                option_positions.append({
                    'asset_idx': i,
                    'type': 'put',
                    'contracts': n_contracts,
                    'strike': strike,
                    'spot_at_sale': spot
                })
                num_contracts += n_contracts
    
    return {
        'premium_collected': premium_collected,
        'option_positions': option_positions,
        'num_contracts': num_contracts
    }


def settle_options(option_positions: list[dict], expiry_prices: np.ndarray,
                   contract_fee: float, spread_bps: float) -> float:
    """
    Cash-settle expired options
    
    Returns net cash flow (premium already collected, this is the settlement cost)
    """
    total_settlement = 0.0
    
    for position in option_positions:
        asset_idx = position['asset_idx']
        strike = position['strike']
        n_contracts = position['contracts']
        expiry_price = expiry_prices[asset_idx]
        
        if position['type'] == 'call':
            # Intrinsic value of call at expiry
            intrinsic = max(0, expiry_price - strike)
        else:  # put
            intrinsic = max(0, strike - expiry_price)
        
        if intrinsic > 0:
            # We have to buy back at intrinsic value
            gross_cost = intrinsic * n_contracts * SHARES_PER_CONTRACT
            
            # Transaction costs for closing
            total_contract_fees = n_contracts * contract_fee
            spread_cost = gross_cost * (spread_bps / 10000)
            
            total_cost = gross_cost + total_contract_fees + spread_cost
            total_settlement += total_cost
    
    # Return negative (cost to us)
    return -total_settlement
