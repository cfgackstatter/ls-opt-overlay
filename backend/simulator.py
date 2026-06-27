import numpy as np
from backend.models import SimulationParams, SimulationResult, StrategyResult, BenchmarkResult
from backend.strategy import run_strategy

# Constants
MIN_STOCK_PRICE = 5.0
MAX_STOCK_PRICE = 200.0
ASSET_VOL_SCALING_MIN = 0.5
ASSET_VOL_SCALING_MAX = 1.5
MONTHS_PER_YEAR = 12
EPSILON = 1e-8


def generate_ar1_factors(
    n_periods: int,
    n_assets: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate autocorrelated factor scores using AR(1) process.
    Pure NumPy — no Numba JIT overhead, RNG is caller-supplied for isolation.
    """
    innovation_std = np.sqrt(1 - rho ** 2)
    factors = np.zeros((n_periods, n_assets))
    factors[0] = rng.standard_normal(n_assets)
    for t in range(1, n_periods):
        factors[t] = rho * factors[t - 1] + rng.standard_normal(n_assets) * innovation_std
    return factors


def generate_correlated_returns(
    factor_scores: np.ndarray,
    ic: float,
    mean: float,
    asset_vols: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate returns correlated with factor scores via IC mixing.
    Fully vectorized — no per-period loop.
    """
    factor_std_scores = (factor_scores - factor_scores.mean()) / (factor_scores.std() + EPSILON)
    noise = rng.standard_normal(factor_scores.shape)
    return mean + asset_vols * (ic * factor_std_scores + np.sqrt(1 - ic ** 2) * noise)


def generate_prices(returns: np.ndarray, initial_prices: np.ndarray) -> np.ndarray:
    """
    Vectorized price generation from returns.

    Args:
        returns: (n_periods, n_assets) return matrix
        initial_prices: (n_assets,) starting prices

    Returns:
        (n_periods + 1, n_assets) prices including t=0
    """
    prices = np.empty((len(returns) + 1, len(initial_prices)))
    prices[0] = initial_prices
    prices[1:] = initial_prices * np.cumprod(1 + returns, axis=0)
    return prices


def compute_benchmark_returns(
    returns: np.ndarray,
    strategy_params: dict[str, float],
    financing_params: dict[str, float],
) -> np.ndarray:
    """
    Equal-weight index plus cash benchmark in decimal monthly returns.
    Vectorized — replaces the original per-period Python loop.
    """
    lookback = int(strategy_params["lookback"])
    length = int(strategy_params["strategy_length"])
    net_exposure = strategy_params["long_weight"] - strategy_params["short_weight"]
    cash_exposure = 1.0 - net_exposure
    monthly_cash_rate = financing_params["cash_rate"] / MONTHS_PER_YEAR

    end = min(lookback + length, returns.shape[0])
    ew_returns = returns[lookback:end].mean(axis=1)  # (length,) equal-weight return per period
    return net_exposure * ew_returns + cash_exposure * monthly_cash_rate


def run_simulation(params: SimulationParams, seed: int = 42) -> SimulationResult:
    """
    Simulate market data and run base + overlay strategies on the same price path.

    Uses np.random.default_rng(seed) — isolated per-simulation, safe for
    parallel execution via ProcessPoolExecutor (no global seed mutation).
    """
    rng = np.random.default_rng(seed)  # isolated RNG — replaces np.random.seed(seed)

    universe = params.universe
    strategy = params.strategy
    financing = params.financing

    n_periods = strategy.lookback + strategy.strategy_length
    monthly_mean = universe.mean_return / MONTHS_PER_YEAR
    monthly_vol = universe.volatility / np.sqrt(MONTHS_PER_YEAR)

    factor_scores: np.ndarray = generate_ar1_factors(
        n_periods, universe.n_assets, universe.factor_autocorr, rng
    )
    asset_vols: np.ndarray = rng.uniform(
        ASSET_VOL_SCALING_MIN * monthly_vol,
        ASSET_VOL_SCALING_MAX * monthly_vol,
        universe.n_assets,
    )
    returns: np.ndarray = generate_correlated_returns(
        factor_scores, universe.ic, monthly_mean, asset_vols, rng
    )
    initial_prices: np.ndarray = rng.uniform(MIN_STOCK_PRICE, MAX_STOCK_PRICE, universe.n_assets)
    prices: np.ndarray = generate_prices(returns, initial_prices)

    strategy_params: dict[str, float] = {
        "lookback": strategy.lookback,
        "strategy_length": strategy.strategy_length,
        "risk_aversion": strategy.risk_aversion,
        "long_weight": strategy.long_weight,
        "short_weight": strategy.short_weight,
        "turnover_limit": strategy.turnover_limit,
        "max_weight": strategy.max_weight,
        "transaction_cost_bps": strategy.transaction_cost_bps,
    }
    financing_params: dict[str, float] = {
        "cash_rate": financing.cash_rate,
        "margin_rate": financing.margin_rate,
        "borrow_fee": financing.borrow_fee,
    }
    # Base options dict with overlay disabled; reused for both runs
    options_base: dict = {
        "enabled": False,
        "call_otm_pct": params.options.call_otm_pct,
        "put_otm_pct": params.options.put_otm_pct,
        "call_alpha_barrier": params.options.call_alpha_barrier,
        "put_alpha_barrier": params.options.put_alpha_barrier,
        "contract_fee": params.options.contract_fee,
        "spread_bps": params.options.spread_bps,
    }

    # Run both strategy variants on the same generated path
    base_dict: dict = run_strategy(
        returns=returns, prices=prices, factor_scores=factor_scores,
        strategy_params=strategy_params, financing_params=financing_params,
        options_params=options_base, verbose=False, log_file=None,
    )
    overlay_dict: dict = run_strategy(
        returns=returns, prices=prices, factor_scores=factor_scores,
        strategy_params=strategy_params, financing_params=financing_params,
        options_params={**options_base, "enabled": True}, verbose=False, log_file=None,
    )

    base_result = StrategyResult(**base_dict)
    overlay_result = StrategyResult(**overlay_dict)

    benchmark_array: np.ndarray = compute_benchmark_returns(
        returns, strategy_params, financing_params
    )

    # Benchmark summary stats
    cum_bench = float(np.prod(1 + benchmark_array) - 1)
    n_b = len(benchmark_array)
    ann_bench = float((1 + cum_bench) ** (12 / n_b) - 1) if n_b > 0 else 0.0
    mean_b, std_b = float(benchmark_array.mean()), float(benchmark_array.std())
    sharpe_b = float(mean_b / std_b * np.sqrt(12)) if std_b > 0 else 0.0

    benchmark_result = BenchmarkResult(
        returns=(benchmark_array * 100).tolist(),
        cumulative_return=cum_bench * 100,
        annualized_return=ann_bench * 100,
        sharpe_ratio=sharpe_b,
    )

    # Align all three return series to the same length before computing active returns
    base_rets = np.array(base_result.portfolio_returns) / 100.0
    overlay_rets = np.array(overlay_result.portfolio_returns) / 100.0
    min_len = min(len(base_rets), len(overlay_rets), len(benchmark_array))
    base_rets = base_rets[:min_len]
    overlay_rets = overlay_rets[:min_len]
    bench = benchmark_array[:min_len]

    active_base = base_rets - bench
    active_overlay = overlay_rets - bench

    ann_alpha_base = float(active_base.mean() * 12 * 100)
    ann_alpha_overlay = float(active_overlay.mean() * 12 * 100)
    te_base = float(active_base.std() * np.sqrt(12) * 100)
    te_overlay = float(active_overlay.std() * np.sqrt(12) * 100)

    return SimulationResult(
        base=base_result,
        with_options=overlay_result,
        benchmark=benchmark_result,
        alpha_base=ann_alpha_base,
        alpha_with_options=ann_alpha_overlay,
        information_ratio_base=ann_alpha_base / te_base if te_base > 0 else 0.0,
        information_ratio_with_options=ann_alpha_overlay / te_overlay if te_overlay > 0 else 0.0,
        options_lift=float(overlay_result.annualized_return - base_result.annualized_return),
    )