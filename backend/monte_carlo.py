# monte_carlo.py
"""Parallel Monte Carlo evaluation of base equity vs options-overlay strategies.

Each draw:
  1. Spawns an independent RNG stream (SeedSequence) for the market DGP
  2. Runs base (no options) and overlay on the *same* price path
  3. Records performance metrics and the cumulative options-lift path

Aggregation uses sample statistics (ddof=1) and reports diagnostics for
strategy validity: P(lift > 0), standard error, and a simple t-statistic.
Failed workers are counted rather than silently dropped from the request size.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, TypedDict

import numpy as np

from backend.models import (
    SimulationParams,
    MonteCarloParams,
    MonteCarloResult,
    MonteCarloDistribution,
    MonteCarloPaths,
    MonteCarloDiagnostics,
)
from backend.simulator import run_simulation


class _Sample(TypedDict):
    metrics: Dict[str, float]
    options_lift_path: List[float]
    n_months: int
    full_horizon: bool


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
    "avg_turnover_base",
    "avg_turnover_with_options",
    "avg_options_income",
]


def _compute_cum_returns(returns_pct: List[float]) -> np.ndarray:
    """Cumulative return series from monthly % returns, starting at 0."""
    r = np.asarray(returns_pct, dtype=float) / 100.0
    if len(r) == 0:
        return np.array([0.0])
    cum = np.cumprod(1.0 + r) - 1.0
    return np.concatenate([[0.0], cum])


def _max_dd(rets_pct: list[float]) -> float:
    rets = np.asarray(rets_pct, dtype=float) / 100.0
    if len(rets) == 0:
        return 0.0
    cum = np.cumprod(1.0 + rets)
    return float((cum / np.maximum.accumulate(cum) - 1.0).min() * 100.0)


def _run_single_sim(args: Tuple[Dict, int, int]) -> Dict[str, Any]:
    """
    Worker: one simulation with an independent seed.

    Returns {"ok": True, "sample": ...} or {"ok": False, "seed": ..., "error": ...}.
    """
    params_dict, seed, target_months = args
    try:
        params = SimulationParams.model_validate(params_dict)
        sim = run_simulation(params, seed=seed)

        base = sim.base
        overlay = sim.with_options
        bench = sim.benchmark

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
            "avg_turnover_base": base.avg_turnover,
            "avg_turnover_with_options": overlay.avg_turnover,
            "avg_options_income": overlay.avg_options_income,
        }

        base_cum = _compute_cum_returns(base.portfolio_returns)
        overlay_cum = _compute_cum_returns(overlay.portfolio_returns)
        n = min(len(base_cum), len(overlay_cum))
        lift_path = ((overlay_cum[:n] - base_cum[:n]) * 100.0).tolist()
        # Months of strategy returns (cum path includes a leading 0)
        n_months = max(n - 1, 0)

        return {
            "ok": True,
            "sample": {
                "metrics": metrics,
                "options_lift_path": lift_path,
                "n_months": n_months,
                "full_horizon": n_months >= target_months,
            },
        }
    except Exception as e:
        return {
            "ok": False,
            "seed": seed,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=5),
        }


def _distribution(values: np.ndarray) -> MonteCarloDistribution:
    values = np.asarray(values, dtype=float)
    p5, p25, p75, p95 = np.percentile(values, [5, 25, 75, 95])
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return MonteCarloDistribution(
        mean=float(values.mean()),
        median=float(np.median(values)),
        std=std,
        p5=float(p5),
        p25=float(p25),
        p75=float(p75),
        p95=float(p95),
        min=float(values.min()),
        max=float(values.max()),
    )


def _pad_paths(paths: List[List[float]], length: int) -> List[List[float]]:
    """Pad shorter paths by holding the terminal lift (e.g. after ruin)."""
    out: List[List[float]] = []
    for p in paths:
        if len(p) >= length:
            out.append(p[:length])
        elif len(p) == 0:
            out.append([0.0] * length)
        else:
            out.append(p + [p[-1]] * (length - len(p)))
    return out


def _lift_diagnostics(lifts: np.ndarray) -> MonteCarloDiagnostics:
    n = int(len(lifts))
    mean = float(lifts.mean()) if n else 0.0
    std = float(lifts.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    tstat = mean / se if se > 0 else 0.0
    return MonteCarloDiagnostics(
        lift_mean=mean,
        lift_std=std,
        lift_se=float(se),
        lift_tstat=float(tstat),
        lift_prob_positive=float(np.mean(lifts > 0.0)) if n else 0.0,
        lift_prob_negative=float(np.mean(lifts < 0.0)) if n else 0.0,
    )


def run_monte_carlo(mc_params: MonteCarloParams) -> MonteCarloResult:
    """
    Run many independent simulations in parallel and aggregate metrics.

    Seeds: ``SeedSequence(base_seed).generate_state(n_sims)`` gives
    statistically independent streams under parallel execution (better than
    ``base_seed + i`` alone).
    """
    n_sims = int(mc_params.n_sims)
    base_seed = int(mc_params.base_seed)
    n_workers = mc_params.n_workers or min(n_sims, os.cpu_count() or 1)
    n_workers = max(1, int(n_workers))

    params_dict = mc_params.simulation_params.model_dump()
    target_months = int(mc_params.simulation_params.strategy.strategy_length)

    # Independent RNG streams for each worker
    seeds = np.random.SeedSequence(base_seed).generate_state(n_sims, dtype=np.uint32)
    tasks: List[Tuple[Dict, int, int]] = [
        (params_dict, int(seeds[i]), target_months) for i in range(n_sims)
    ]

    t0 = time.perf_counter()
    samples: List[_Sample] = []
    failures: List[str] = []

    # maxtasksperschild limits CVXPY / OSQP memory growth (Python ≥3.11)
    pool_kwargs: Dict[str, Any] = {"max_workers": n_workers}
    if sys.version_info >= (3, 11):
        pool_kwargs["max_tasks_per_child"] = 32

    with ProcessPoolExecutor(**pool_kwargs) as executor:
        futures = {executor.submit(_run_single_sim, task): task[1] for task in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            if result.get("ok"):
                samples.append(result["sample"])  # type: ignore[arg-type]
            else:
                msg = f"seed={result.get('seed')}: {result.get('error')}"
                failures.append(msg)
                print(f"[MC] Simulation failed — {msg}")

    runtime = time.perf_counter() - t0

    if not samples:
        detail = "; ".join(failures[:5]) if failures else "unknown"
        raise RuntimeError(f"All Monte Carlo simulations failed. Examples: {detail}")

    metric_samples = [s["metrics"] for s in samples]
    raw: Dict[str, List[float]] = {
        m: [ms[m] for ms in metric_samples] for m in _METRICS
    }
    distributions = {m: _distribution(np.asarray(values)) for m, values in raw.items()}

    lift_paths = [s["options_lift_path"] for s in samples]
    max_len = max(len(p) for p in lift_paths)
    # Prefer target horizon (+1 for leading zero) when any sim reached it
    target_path_len = target_months + 1
    path_len = max(max_len, 1)
    # Cap display at target+1 unless somehow longer
    if max_len >= target_path_len:
        path_len = target_path_len
    lift_paths = _pad_paths(lift_paths, path_len)
    months = list(range(path_len))

    lifts = np.asarray(raw["options_lift"], dtype=float)
    diagnostics = _lift_diagnostics(lifts)
    n_full = sum(1 for s in samples if s["full_horizon"])

    return MonteCarloResult(
        distributions=distributions,
        raw=raw,
        paths=MonteCarloPaths(months=months, options_lift_paths=lift_paths),
        diagnostics=diagnostics,
        n_sims_requested=n_sims,
        n_sims_completed=len(samples),
        n_sims_failed=len(failures),
        n_sims_full_horizon=n_full,
        runtime_seconds=round(runtime, 2),
        failure_messages=failures[:20],
    )
