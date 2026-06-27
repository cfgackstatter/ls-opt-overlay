import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, TypedDict

import numpy as np

from backend.models import (
    SimulationParams,
    MonteCarloParams,
    MonteCarloResult,
    MonteCarloDistribution,
    MonteCarloPaths,
)
from backend.simulator import run_simulation

class _Sample(TypedDict):
    metrics: Dict[str, float]
    options_lift_path: List[float]

# Metrics to collect per simulation
_METRICS = [
    "annualized_return_base",
    "annualized_return_with_options",
    "annualized_return_benchmark",
    "options_lift",
    "alpha_base",
    "alpha_with_options",
    "information_ratio_base",
    "information_ratio_with_options",
    "sharpe_base",
    "sharpe_with_options",
    "max_drawdown_base",
    "max_drawdown_with_options",
]

def _compute_cum_returns(returns_pct: List[float]) -> np.ndarray:
    """Cumulative return series from monthly % returns."""
    r = np.array(returns_pct) / 100.0
    if len(r) == 0:
        return np.array([0.0])
    cum = np.cumprod(1 + r) - 1
    return np.concatenate([[0.0], cum])

def _max_dd(rets_pct: list[float]) -> float:
    rets = np.asarray(rets_pct) / 100.0
    if not len(rets): return 0.0
    cum = np.cumprod(1 + rets)
    return float((cum / np.maximum.accumulate(cum) - 1).min() * 100.0)

def _run_single_sim(args: Tuple[Dict, int]) -> Dict[str, object] | None:
    """
    Worker: one simulation with a unique seed.
    Returns both scalar metrics and the options lift path.
    """
    params_dict, seed = args
    try:
        params = SimulationParams.model_validate(params_dict)
        sim = run_simulation(params, seed=seed)

        base = sim.base
        overlay = sim.with_options
        bench = sim.benchmark


        # Scalar metrics
        metrics = {
            "annualized_return_base": base.annualized_return,
            "annualized_return_with_options": overlay.annualized_return,
            "annualized_return_benchmark": bench.annualized_return,
            "options_lift": sim.options_lift,
            "alpha_base": sim.alpha_base,
            "alpha_with_options": sim.alpha_with_options,
            "information_ratio_base": sim.information_ratio_base,
            "information_ratio_with_options": sim.information_ratio_with_options,
            "sharpe_base": base.sharpe_ratio,
            "sharpe_with_options": overlay.sharpe_ratio,
            "max_drawdown_base": _max_dd(base.portfolio_returns),
            "max_drawdown_with_options": _max_dd(overlay.portfolio_returns),
        }

        # Options lift path: cumulative (overlay – base)
        base_cum = _compute_cum_returns(base.portfolio_returns)
        overlay_cum = _compute_cum_returns(overlay.portfolio_returns)
        n = min(len(base_cum), len(overlay_cum))
        lift_path = (overlay_cum[:n] - base_cum[:n]) * 100.0  # %

        return {
            "metrics": metrics,
            "options_lift_path": lift_path.tolist(),
        }

    except Exception as e:
        print(f"[MC] Simulation with seed {seed} failed: {e}")
        return None

def _distribution(values: np.ndarray) -> MonteCarloDistribution:
    p5, p25, p75, p95 = np.percentile(values, [5, 25, 75, 95])  # single call
    return MonteCarloDistribution(
        mean=float(values.mean()), median=float(np.median(values)),
        std=float(values.std()), p5=float(p5), p25=float(p25),
        p75=float(p75), p95=float(p95),
        min=float(values.min()), max=float(values.max()),
    )

def run_monte_carlo(mc_params: MonteCarloParams) -> MonteCarloResult:
    """
    Run many independent simulations in parallel and aggregate metrics
    and options-lift paths.
    """
    n_sims = mc_params.n_sims
    base_seed = mc_params.base_seed
    n_workers = mc_params.n_workers or min(n_sims, os.cpu_count() or 1)

    params_dict = mc_params.simulation_params.model_dump()
    tasks: List[Tuple[Dict, int]] = [
        (params_dict, base_seed + i) for i in range(n_sims)
    ]

    t0 = time.perf_counter()
    samples: List[_Sample] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_single_sim, task): task for task in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                samples.append(result) # type: ignore[arg-type]

    runtime = time.perf_counter() - t0

    if not samples:
        raise RuntimeError("All Monte Carlo simulations failed.")

    # Aggregate scalar metrics
    metric_samples: List[Dict[str, float]] = [s["metrics"] for s in samples]
    raw: Dict[str, List[float]] = {
        m: [ms[m] for ms in metric_samples] for m in _METRICS
    }
    distributions: Dict[str, MonteCarloDistribution] = {
        m: _distribution(np.array(values)) for m, values in raw.items()
    }

    # Aggregate options lift paths
    lift_paths: List[List[float]] = [s["options_lift_path"] for s in samples]
    # Normalise lengths to the minimum horizon so alignment is easy
    min_len = min(len(p) for p in lift_paths)
    lift_paths = [p[:min_len] for p in lift_paths]
    months = list(range(min_len))  # 0..T, already with 0 at start

    paths = MonteCarloPaths(
        months=months,
        options_lift_paths=lift_paths,
    )

    return MonteCarloResult(
        distributions=distributions,
        raw=raw,
        paths=paths,
        n_sims_completed=len(samples),
        runtime_seconds=round(runtime, 2),
    )
