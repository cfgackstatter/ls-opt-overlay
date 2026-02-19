from typing import Optional, Dict, List
from pydantic import BaseModel, Field

class UniverseParams(BaseModel):
    """Parameters for the underlying asset universe"""
    n_assets: int = 50
    mean_return: float = 6.0
    volatility: float = 20.0
    ic: float = 0.1
    factor_autocorr: float = 0.8

class FinancingParams(BaseModel):
    """Financing cost parameters"""
    cash_rate: float = 3.0
    margin_rate: float = 5.0
    borrow_fee: float = 1.0

class StrategyParams(BaseModel):
    """Parameters for the long-short strategy"""
    strategy_length: int = 60
    risk_aversion: float = 4.0
    long_weight: float = 100.0
    short_weight: float = 100.0
    turnover_limit: float = 15.0
    max_weight: float = 10.0
    lookback: int = 36
    transaction_cost_bps: float = 5.0

class OptionsParams(BaseModel):
    """Options overlay parameters"""
    enabled: bool = False
    call_otm_pct: float = 0.0  # % above spot for call strike
    put_otm_pct: float = 0.0   # % below spot for put strike
    call_alpha_barrier: float = -0.1  # Sell calls if alpha < barrier
    put_alpha_barrier: float = 0.1   # Sell puts if alpha > barrier (for shorts)
    contract_fee: float = 0.65  # Fixed fee per contract
    spread_bps: float = 100.0    # Bid-ask spread cost in basis points

class SimulationParams(BaseModel):
    """Combined simulation parameters"""
    universe: UniverseParams = UniverseParams()
    financing: FinancingParams = FinancingParams()
    strategy: StrategyParams = StrategyParams()
    options: OptionsParams = OptionsParams()

class StrategyResult(BaseModel):
    """Results from strategy backtest (single portfolio stream)"""
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
    """Benchmark performance stats using the same return path"""
    returns: list[float]          # monthly returns in percent
    cumulative_return: float      # in percent
    annualized_return: float      # in percent
    sharpe_ratio: float           # optional, if you want

class SimulationResult(BaseModel):
    """Complete simulation results: base, overlay, and benchmark on same prices"""
    # Three comparable portfolios
    base: StrategyResult              # no options
    with_options: StrategyResult      # with options overlay
    benchmark: BenchmarkResult

    # Relative performance
    alpha_base: float                 # annualized, % vs benchmark
    alpha_with_options: float         # ditto
    information_ratio_base: float
    information_ratio_with_options: float
    options_lift: float               # annualized_return(with_options) - annualized_return(base)

class MonteCarloParams(BaseModel):
    """Parameters for a Monte Carlo run."""
    simulation_params: SimulationParams
    n_sims: int = Field(default=10, ge=1, le=2000)
    base_seed: int = 42
    n_workers: Optional[int] = None  # None -> use cpu_count()

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
    """
    Time‑series output for Monte Carlo.
    Each entry in options_lift_paths is one simulation's path:
    a list of cumulative options lift (%) by month.
    """
    months: List[int]                # [0, 1, 2, ..., T]
    options_lift_paths: List[List[float]]

class MonteCarloResult(BaseModel):
    """Monte Carlo output: distributions + raw arrays + time‑series paths."""
    distributions: Dict[str, MonteCarloDistribution]
    raw: Dict[str, List[float]]
    paths: MonteCarloPaths
    n_sims_completed: int
    runtime_seconds: float