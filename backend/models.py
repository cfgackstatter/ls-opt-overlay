from typing import Optional
from pydantic import BaseModel, Field


class UniverseParams(BaseModel):
    """Parameters for the synthetic equity universe. Rates in annualized decimals."""
    n_assets: int = 50
    mean_return: float = 0.065      # ~6.5% total equity drift (realistic single-name GBM)
    volatility: float = 0.30        # ~30% single-stock vol (idio backed out after factors)
    ic: float = 0.14                # residual IC — optimistic but within research range
    factor_autocorr: float = 0.75   # AR(1) persistence of the alpha signal
    # Common-factor / risk structure
    market_vol: float = 0.16        # ~16% equity-index factor vol
    market_autocorr: float = 0.05
    style_vol: float = 0.07         # size/value factor vol
    style_autocorr: float = 0.10
    avg_beta: float = 1.0
    beta_dispersion: float = 0.30
    # Residual distribution & stochastic vol
    student_t_df: float = 6.0       # moderate fat tails
    stoch_vol_persistence: float = 0.85
    stoch_vol_of_vol: float = 0.25


class FinancingParams(BaseModel):
    """Financing cost parameters. All rates in annualized decimals."""
    cash_rate: float = 0.045        # ~4.5% cash / SOFR-like
    margin_rate: float = 0.065      # cash + ~200 bps
    borrow_fee: float = 0.01        # ~100 bps GC stock-loan


class StrategyParams(BaseModel):
    """
    Portfolio construction parameters.

    Sleeve exposures (decimals of NAV):
      - long_only 100%:  long_weight=1.0, short_weight=0.0
      - 130/30:          long_weight=1.3, short_weight=0.3
      - market-neutral:  long_weight=1.0, short_weight=1.0
      - levered long:    long_weight=1.5, short_weight=0.0  (cash residual = -0.5)

    Optimizer objective (standard):
      max α'w − λ w'Σw − κ₁‖Δw‖₁ − κ₂‖Δw‖₂²
    where κ₁ = transaction_cost_bps / 1e4 and κ₂ = market_impact_coef.
    Optional hard_turnover_limit (>0) adds Σ|Δw| ≤ limit; 0 disables it.

    Defaults favour sticky books (material κ₁) with informative Grinold alphas so
    the options overwrite has low-conviction names to work on.
    """
    strategy_length: int = 60
    risk_aversion: float = 1.0      # active book; pairs with 200/200 gross
    long_weight: float = 2.0        # levered market-neutral (200/200)
    short_weight: float = 2.0       # net 0, gross 4 — symmetric call/put overwrite
    max_long_weight: float = 0.08   # 50×0.08 ≥ 2.0 sleeve
    max_short_weight: float = 0.08
    lookback: int = 36
    # Transaction costs in the objective (and matched in cash accounting)
    transaction_cost_bps: float = 30.0  # sticky turnover → overwrite opportunities
    market_impact_coef: float = 0.0
    hard_turnover_limit: float = 0.0
    # Deprecated alias kept for older notebooks / API clients
    turnover_limit: float = 0.0
    weight_threshold: float = 1e-4
    # Alpha / risk model — signal IC near DGP IC so barriers are informative
    signal_ic: float = 0.12
    alpha_method: str = "grinold"    # grinold | rank_grinold | zscore
    cov_method: str = "ewma"         # ewma | ledoit
    cov_halflife: int = 24


class OptionsParams(BaseModel):
    """
    Sticky-alpha overwrite overlay.

    Sell 1M covered calls on longs with α ≤ call_alpha_barrier;
    sell 1M covered puts on shorts with α ≥ put_alpha_barrier.
    Prices are European BSM mids (risk-model vol); trade at half-spread
    from mid; settle at intrinsic. dividend_yield should stay 0 unless the
    equity DGP pays dividends.

    Defaults: mild OTM, slightly selective barriers, tight costs — set so
    alpha-conditioned overwrite has a realistic chance of positive lift.
    """
    enabled: bool = True
    call_otm_pct: float = 0.0           # ATM overwrite — more premium; still 1M covered
    put_otm_pct: float = 0.0
    call_alpha_barrier: float = -0.0015 # overwrite only clearly weak longs (~−0.15%/mo)
    put_alpha_barrier: float = 0.0015   # overwrite only clearly unwanted shorts
    contract_fee: float = 0.40          # $ per contract (open + ITM exercise)
    spread_bps: float = 10.0            # liquid single-name-ish full spread
    dividend_yield: float = 0.0         # BSM q; 0 matches non-dividend GBM


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


class MonteCarloDiagnostics(BaseModel):
    """Validity diagnostics focused on options lift across simulations."""
    lift_mean: float
    lift_std: float
    lift_se: float          # standard error of the mean
    lift_tstat: float       # mean / se (large-sample)
    lift_prob_positive: float
    lift_prob_negative: float


class MonteCarloResult(BaseModel):
    """Monte Carlo output: distributions + raw arrays + time-series paths."""
    distributions: dict[str, MonteCarloDistribution]
    raw: dict[str, list[float]]
    paths: MonteCarloPaths
    diagnostics: MonteCarloDiagnostics
    n_sims_requested: int
    n_sims_completed: int
    n_sims_failed: int = 0
    n_sims_full_horizon: int = 0
    runtime_seconds: float
    failure_messages: list[str] = []

