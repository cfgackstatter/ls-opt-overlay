import numpy as np
from numba import jit
from backend.models import SimulationParams, SimulationResult, StrategyResult, BenchmarkResult
from backend.strategy import run_strategy

# Constants
MIN_STOCK_PRICE = 5.0
MAX_STOCK_PRICE = 200.0
ASSET_VOL_SCALING_MIN = 0.5
ASSET_VOL_SCALING_MAX = 1.5
MONTHS_PER_YEAR = 12
EPSILON = 1e-8


def compute_benchmark_returns(
    returns: np.ndarray,
    strategy_params: dict,
    financing_params: dict,
) -> np.ndarray:
    """
    Equal-weight index plus cash benchmark in decimal monthly returns.
    """
    lookback = int(strategy_params['lookback'])
    length = int(strategy_params['strategy_length'])

    net_exposure = (
        strategy_params['long_weight'] - strategy_params['short_weight']
    ) / 100.0
    cash_exposure = 1.0 - net_exposure
    monthly_cash_rate = (financing_params['cash_rate'] / 100.0) / MONTHS_PER_YEAR

    n_periods, _ = returns.shape
    benchmark = []

    for t in range(lookback + 1, lookback + length + 1):
        if t >= n_periods + 1:
            break
        equal_weight_return = np.mean(returns[t - 1])          # same as before
        br = net_exposure * equal_weight_return + cash_exposure * monthly_cash_rate
        benchmark.append(br)

    return np.array(benchmark)


def run_simulation(params: SimulationParams, seed: int = 42) -> SimulationResult:
    """
    Simulate market data and run the strategy:
    - base portfolio (no options)
    - options overlay portfolio
    - benchmark on the same returns
    """
    np.random.seed(seed)

    universe = params.universe
    strategy = params.strategy
    financing = params.financing
    
    n_periods = strategy.lookback + strategy.strategy_length
    monthly_mean = (universe.mean_return / 100) / MONTHS_PER_YEAR
    monthly_vol = (universe.volatility / 100) / np.sqrt(MONTHS_PER_YEAR)
    
    factor_scores = generate_ar1_factors_numba(
        n_periods,
        universe.n_assets,
        universe.factor_autocorr,
        seed,
    )
    
    asset_vols = np.random.uniform(
        ASSET_VOL_SCALING_MIN * monthly_vol,
        ASSET_VOL_SCALING_MAX * monthly_vol,
        universe.n_assets,
    )
    
    returns = generate_correlated_returns_numba(
        factor_scores,
        universe.ic,
        monthly_mean,
        asset_vols,
        seed + 1,
    )
    
    initial_prices = np.random.uniform(MIN_STOCK_PRICE, MAX_STOCK_PRICE, universe.n_assets)
    prices = generate_prices(returns, initial_prices)
    
    strategy_params = {
        'lookback': strategy.lookback,
        'strategy_length': strategy.strategy_length,
        'risk_aversion': strategy.risk_aversion,
        'long_weight': strategy.long_weight,
        'short_weight': strategy.short_weight,
        'turnover_limit': strategy.turnover_limit,
        'max_weight': strategy.max_weight,
        'transaction_cost_bps': strategy.transaction_cost_bps,
    }

    financing_params = {
        'cash_rate': financing.cash_rate,
        'margin_rate': financing.margin_rate,
        'borrow_fee': financing.borrow_fee,
    }
    
    # BASE STRATEGY: options disabled
    options_params_base = {
        'enabled': False,
        'call_otm_pct': params.options.call_otm_pct,
        'put_otm_pct': params.options.put_otm_pct,
        'call_alpha_barrier': params.options.call_alpha_barrier,
        'put_alpha_barrier': params.options.put_alpha_barrier,
        'contract_fee': params.options.contract_fee,
        'spread_bps': params.options.spread_bps,
    }

    base_dict = run_strategy(
        returns=returns,
        prices=prices,
        factor_scores=factor_scores,
        strategy_params=strategy_params,
        financing_params=financing_params,
        options_params=options_params_base,
        verbose=False,
        log_file=None,
    )

    # WITH OPTIONS OVERLAY: options enabled (only if user wants overlay)
    options_params_overlay = dict(options_params_base)
    options_params_overlay['enabled'] = True

    with_overlay_dict = run_strategy(
        returns=returns,
        prices=prices,
        factor_scores=factor_scores,
        strategy_params=strategy_params,
        financing_params=financing_params,
        options_params=options_params_overlay,
        verbose=False,
        log_file=None,
    )

    base_result = StrategyResult(**base_dict)
    with_options_result = StrategyResult(**with_overlay_dict)

    benchmark_array = compute_benchmark_returns(returns, strategy_params, financing_params)
    # convert to percent for the model
    benchmark_pct = benchmark_array * 100.0

    # summary stats
    cum_bench = np.prod(1 + benchmark_array) - 1
    n_months_b = len(benchmark_array)
    ann_bench = (
        (1 + cum_bench) ** (12 / n_months_b) - 1
        if n_months_b > 0 else 0.0
    )
    mean_b = np.mean(benchmark_array) if n_months_b > 0 else 0.0
    std_b = np.std(benchmark_array) if n_months_b > 0 else 0.0
    sharpe_b = (mean_b / std_b * np.sqrt(12)) if std_b > 0 else 0.0

    benchmark_result = BenchmarkResult(
        returns=[float(r) for r in benchmark_pct],
        cumulative_return=float(cum_bench * 100),
        annualized_return=float(ann_bench * 100),
        sharpe_ratio=float(sharpe_b),
    )

    base_rets = np.array(base_result.portfolio_returns) / 100.0
    overlay_rets = np.array(with_options_result.portfolio_returns) / 100.0

    # Align lengths (in case of early stop)
    min_len = min(len(base_rets), len(overlay_rets), len(benchmark_array))
    base_rets = base_rets[:min_len]
    overlay_rets = overlay_rets[:min_len]
    bench = benchmark_array[:min_len]

    active_base = base_rets - bench
    active_overlay = overlay_rets - bench

    ann_alpha_base = float(np.mean(active_base) * 12 * 100)
    ann_alpha_overlay = float(np.mean(active_overlay) * 12 * 100)

    te_base = float(np.std(active_base) * np.sqrt(12) * 100)
    te_overlay = float(np.std(active_overlay) * np.sqrt(12) * 100)

    ir_base = float((ann_alpha_base / te_base) if te_base > 0 else 0.0)
    ir_overlay = float((ann_alpha_overlay / te_overlay) if te_overlay > 0 else 0.0)

    options_lift = (
        with_options_result.annualized_return - base_result.annualized_return
    )
    
    sim_result = SimulationResult(
        base=base_result,
        with_options=with_options_result,
        benchmark=benchmark_result,
        alpha_base=ann_alpha_base,
        alpha_with_options=ann_alpha_overlay,
        information_ratio_base=ir_base,
        information_ratio_with_options=ir_overlay,
        options_lift=float(options_lift),
    )

    return sim_result


def generate_prices(returns: np.ndarray, initial_prices: np.ndarray) -> np.ndarray:
    """
    Vectorized price generation from returns
    
    Args:
        returns: (n_periods, n_assets) returns
        initial_prices: (n_assets,) starting prices
    
    Returns:
        (n_periods + 1, n_assets) prices including initial prices
    """
    prices = np.zeros((len(returns) + 1, len(initial_prices)))
    prices[0] = initial_prices
    prices[1:] = initial_prices * np.cumprod(1 + returns, axis=0)
    return prices


@jit(nopython=True)
def generate_ar1_factors_numba(n_periods: int, n_assets: int, rho: float, seed: int) -> np.ndarray:
    """Generate autocorrelated factor scores using AR(1) process - Numba optimized"""
    np.random.seed(seed)
    factors = np.zeros((n_periods, n_assets))
    factors[0] = np.random.randn(n_assets)
    innovation_std = np.sqrt(1 - rho**2)
    
    for t in range(1, n_periods):
        innovations = np.random.randn(n_assets) * innovation_std
        factors[t] = rho * factors[t-1] + innovations
    
    return factors


@jit(nopython=True)
def generate_correlated_returns_numba(factor_scores: np.ndarray, ic: float,
                                      mean: float, asset_vols: np.ndarray, seed: int) -> np.ndarray:
    """Generate returns correlated with factor scores - Numba optimized"""
    np.random.seed(seed)
    n_periods, n_assets = factor_scores.shape
    
    # Standardize factor scores
    factor_mean = np.mean(factor_scores)
    factor_std = np.std(factor_scores)
    factor_std_scores = (factor_scores - factor_mean) / (factor_std + EPSILON)
    
    # Generate independent noise
    noise = np.random.randn(n_periods, n_assets)
    
    # Combine IC scaling and volatility scaling in one step
    returns = mean + asset_vols[:, np.newaxis].T * (
        ic * factor_std_scores + np.sqrt(1 - ic**2) * noise
    )
    
    return returns
