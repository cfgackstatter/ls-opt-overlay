import numpy as np
from backend.models import SimulationParams, SimulationResult, StrategyResult, BenchmarkResult
from backend.strategy import run_strategy
from backend.optimizer import PortfolioOptimizer
from backend.market import (
    MONTHS_PER_YEAR,
    generate_market_data,
    # Re-export for notebooks / external callers
    generate_ar1_factors,
    generate_correlated_returns,
    generate_prices,
)

__all__ = [
    "MONTHS_PER_YEAR",
    "generate_ar1_factors",
    "generate_correlated_returns",
    "generate_prices",
    "generate_market_data",
    "compute_benchmark_returns",
    "run_simulation",
]


def compute_benchmark_returns(
    returns: np.ndarray,
    strategy_params: dict[str, float],
    financing_params: dict[str, float],
) -> np.ndarray:
    """
    Equal-weight index plus cash benchmark in decimal monthly simple returns.
    Matches the strategy's net equity exposure.
    """
    lookback = int(strategy_params["lookback"])
    length = int(strategy_params["strategy_length"])
    net_exposure = strategy_params["long_weight"] - strategy_params["short_weight"]
    cash_exposure = 1.0 - net_exposure
    monthly_cash_rate = financing_params["cash_rate"] / MONTHS_PER_YEAR

    end = min(lookback + length, returns.shape[0])
    ew_returns = returns[lookback:end].mean(axis=1)
    return net_exposure * ew_returns + cash_exposure * monthly_cash_rate


def run_simulation(params: SimulationParams, seed: int = 42) -> SimulationResult:
    """
    Simulate market data and run base + overlay strategies on the same price path.

    Uses np.random.default_rng(seed) — isolated per-simulation, safe for
    parallel execution via ProcessPoolExecutor.
    """
    rng = np.random.default_rng(seed)

    universe = params.universe
    strategy = params.strategy
    financing = params.financing

    n_periods = strategy.lookback + strategy.strategy_length

    market = generate_market_data(
        n_periods,
        universe.n_assets,
        mean_return=universe.mean_return,
        volatility=universe.volatility,
        ic=universe.ic,
        factor_autocorr=universe.factor_autocorr,
        market_vol=universe.market_vol,
        market_autocorr=universe.market_autocorr,
        style_vol=universe.style_vol,
        style_autocorr=universe.style_autocorr,
        avg_beta=universe.avg_beta,
        beta_dispersion=universe.beta_dispersion,
        student_t_df=universe.student_t_df,
        stoch_vol_persistence=universe.stoch_vol_persistence,
        stoch_vol_of_vol=universe.stoch_vol_of_vol,
        rng=rng,
    )
    returns = market.simple_returns
    prices = market.prices
    factor_scores = market.factor_scores

    strategy_params: dict = {
        "lookback": strategy.lookback,
        "strategy_length": strategy.strategy_length,
        "risk_aversion": strategy.risk_aversion,
        "long_weight": strategy.long_weight,
        "short_weight": strategy.short_weight,
        "max_long_weight": strategy.max_long_weight,
        "max_short_weight": strategy.max_short_weight,
        "transaction_cost_bps": strategy.transaction_cost_bps,
        "market_impact_coef": strategy.market_impact_coef,
        "hard_turnover_limit": strategy.hard_turnover_limit or strategy.turnover_limit,
        "turnover_limit": strategy.turnover_limit,
        "weight_threshold": strategy.weight_threshold,
        "signal_ic": strategy.signal_ic,
        "alpha_method": strategy.alpha_method,
        "cov_method": strategy.cov_method,
        "cov_halflife": strategy.cov_halflife,
    }
    financing_params: dict[str, float] = {
        "cash_rate": financing.cash_rate,
        "margin_rate": financing.margin_rate,
        "borrow_fee": financing.borrow_fee,
    }
    options_base: dict = {
        "enabled": False,
        "call_otm_pct": params.options.call_otm_pct,
        "put_otm_pct": params.options.put_otm_pct,
        "call_alpha_barrier": params.options.call_alpha_barrier,
        "put_alpha_barrier": params.options.put_alpha_barrier,
        "contract_fee": params.options.contract_fee,
        "spread_bps": params.options.spread_bps,
        "dividend_yield": params.options.dividend_yield,
    }

    hard_limit = strategy.hard_turnover_limit or strategy.turnover_limit
    tc_linear = strategy.transaction_cost_bps / 10_000.0
    opt_kwargs = dict(
        n_assets=universe.n_assets,
        risk_aversion=strategy.risk_aversion,
        long_weight=strategy.long_weight,
        short_weight=strategy.short_weight,
        max_long_weight=strategy.max_long_weight,
        max_short_weight=strategy.max_short_weight,
        hard_turnover_limit=hard_limit,
        tc_linear=tc_linear,
        tc_quad=strategy.market_impact_coef,
    )
    # Separate solvers so OSQP warm-start from the base book cannot bias the overlay path
    optimizer_base = PortfolioOptimizer(**opt_kwargs)
    optimizer_overlay = PortfolioOptimizer(**opt_kwargs)

    base_dict: dict = run_strategy(
        returns=returns, prices=prices, factor_scores=factor_scores,
        strategy_params=strategy_params, financing_params=financing_params,
        options_params=options_base, verbose=False, log_file=None,
        optimizer=optimizer_base,
    )
    overlay_dict: dict = run_strategy(
        returns=returns, prices=prices, factor_scores=factor_scores,
        strategy_params=strategy_params, financing_params=financing_params,
        options_params={**options_base, "enabled": params.options.enabled},
        verbose=False, log_file=None,
        optimizer=optimizer_overlay,
    )

    base_result = StrategyResult(**base_dict)
    overlay_result = StrategyResult(**overlay_dict)

    benchmark_array: np.ndarray = compute_benchmark_returns(
        returns, strategy_params, financing_params
    )

    cum_bench = float(np.prod(1 + benchmark_array) - 1)
    n_b = len(benchmark_array)
    if n_b > 0 and (1 + cum_bench) > 0:
        ann_bench = float((1 + cum_bench) ** (12 / n_b) - 1)
    else:
        ann_bench = -1.0 if n_b > 0 else 0.0
    mean_b, std_b = float(benchmark_array.mean()), float(benchmark_array.std())
    sharpe_b = float(mean_b / std_b * np.sqrt(12)) if std_b > 0 else 0.0

    benchmark_result = BenchmarkResult(
        returns=(benchmark_array * 100).tolist(),
        cumulative_return=cum_bench * 100,
        annualized_return=ann_bench * 100,
        sharpe_ratio=sharpe_b,
    )

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
