from typing import Optional
from pydantic import BaseModel, Field


class UniverseParams(BaseModel):
    """Parameters for the underlying asset universe. All rates in decimals (0.06 = 6%)."""
    n_assets: int = 50
    mean_return: float = 0.0       # annualized, e.g. 0.06 = 6%
    volatility: float = 0.20        # annualized, e.g. 0.20 = 20%
    ic: float = 0.12                # factor IC; dimensionless
    factor_autocorr: float = 0.7    # AR(1) persistence; dimensionless


class FinancingParams(BaseModel):
    """Financing cost parameters. All rates in annualized decimals."""
    cash_rate: float = 0.04         # 4% p.a.
    margin_rate: float = 0.055      # 5.5% p.a.
    borrow_fee: float = 0.005       # 0.5% p.a.


class StrategyParams(BaseModel):
    """Parameters for the long-short strategy. Weights/limits in decimals."""
    strategy_length: int = 60       # months
    risk_aversion: float = 2.0      # dimensionless
    long_weight: float = 1.30       # 130% gross long
    short_weight: float = 0.30      # 30% gross short
    turnover_limit: float = 0.20    # 20% one-way per rebalance
    max_weight: float = 0.10        # 10% max single stock
    lookback: int = 36              # months
    transaction_cost_bps: float = 5.0  # bps (not a rate, kept as bps)


class OptionsParams(BaseModel):
    """Options overlay parameters."""
    enabled: bool = True
    call_otm_pct: float = 0.01      # call strike = spot * (1 + 0.01)
    put_otm_pct: float = 0.01       # put  strike = spot * (1 - 0.01)
    call_alpha_barrier: float = -0.001  # monthly decimal (-0.1% per month)
    put_alpha_barrier: float = 0.001    # monthly decimal (+0.1% per month)
    contract_fee: float = 0.65      # $ per contract
    spread_bps: float = 25.0        # bps


class SimulationParams(BaseModel):
    """Combined simulation parameters."""
    universe: UniverseParams = UniverseParams()
    financing: FinancingParams = FinancingParams()
    strategy: StrategyParams = StrategyParams()
    options: OptionsParams = OptionsParams()


# --- Result models below are unchanged ---

class StrategyResult(BaseModel):
    """Results from strategy backtest (single portfolio stream)."""
    portfolio_returns: list[float]
    financing_costs: list[float]
    portfolio_weights: list[list[float]]
    turnovers: list[float]
    cumulative_return: float
    annualized_return: float
    sharpe_ratio: float
    avg_turnover: float
    avg_financing_cost: float
    options_income: list[float]
    avg_options_income: float


class BenchmarkResult(BaseModel):
    """Benchmark performance stats using the same return path."""
    returns: list[float]        # monthly returns in percent
    cumulative_return: float    # in percent
    annualized_return: float    # in percent
    sharpe_ratio: float


class SimulationResult(BaseModel):
    """Complete simulation results: base, overlay, and benchmark on same prices."""
    base: StrategyResult
    with_options: StrategyResult
    benchmark: BenchmarkResult
    alpha_base: float
    alpha_with_options: float
    information_ratio_base: float
    information_ratio_with_options: float
    options_lift: float


class MonteCarloParams(BaseModel):
    """Parameters for a Monte Carlo run."""
    simulation_params: SimulationParams
    base_seed: int = 42
    n_workers: Optional[int] = None
    n_sims: int = Field(default=10, ge=1, le=2000)


class MonteCarloDistribution(BaseModel):
    """Summary statistics for one metric across all simulations."""
    mean: float
    median: float
    std: float
    p5: float
    p25: float
    p75: float
    p95: float
    min: float
    max: float


class MonteCarloPaths(BaseModel):
    months: list[int]
    options_lift_paths: list[list[float]]


class MonteCarloResult(BaseModel):
    """Monte Carlo output: distributions + raw arrays + time-series paths."""
    distributions: dict[str, MonteCarloDistribution]
    raw: dict[str, list[float]]
    paths: MonteCarloPaths
    n_sims_completed: int
    runtime_seconds: float