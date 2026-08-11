const API_BASE = '';

export interface UniverseParams {
  n_assets: number;
  mean_return: number;
  volatility: number;
  ic: number;
  factor_autocorr: number;
  market_vol: number;
  market_autocorr: number;
  style_vol: number;
  style_autocorr: number;
  avg_beta: number;
  beta_dispersion: number;
  student_t_df: number;
  stoch_vol_persistence: number;
  stoch_vol_of_vol: number;
}

export interface FinancingParams {
  cash_rate: number;
  margin_rate: number;
  borrow_fee: number;
}

export interface StrategyParams {
  strategy_length: number;
  risk_aversion: number;
  long_weight: number;
  short_weight: number;
  max_long_weight: number;
  max_short_weight: number;
  lookback: number;
  /** Linear TC in objective (and cash): bps of NAV per unit Σ|Δw| */
  transaction_cost_bps: number;
  /** Quadratic market-impact penalty κ₂ on Σ(Δw²); 0 = off */
  market_impact_coef: number;
  /** Optional hard Σ|Δw| cap; 0 = disabled (preferred) */
  hard_turnover_limit: number;
  /** @deprecated alias for hard_turnover_limit */
  turnover_limit?: number;
  weight_threshold: number;
  signal_ic: number;
  alpha_method: string;
  cov_method: string;
  cov_halflife: number;
}

export interface OptionsParams {
  enabled: boolean;
  call_otm_pct: number;
  put_otm_pct: number;
  call_alpha_barrier: number;
  put_alpha_barrier: number;
  contract_fee: number;
  spread_bps: number;
  dividend_yield: number;
}

export interface SimulationParams {
  universe: UniverseParams;
  financing: FinancingParams;
  strategy: StrategyParams;
  options: OptionsParams;
}

export interface MonteCarloParams {
  simulation_params: SimulationParams;
  n_sims: number;
  base_seed?: number;
  n_workers?: number | null;
}

export interface MonteCarloDistribution {
  mean: number;
  median: number;
  std: number;
  p5: number;
  p25: number;
  p75: number;
  p95: number;
  min: number;
  max: number;
}

export interface MonteCarloPaths {
  months: number[];
  options_lift_paths: number[][];
}

export interface MonteCarloDiagnostics {
  lift_mean: number;
  lift_std: number;
  lift_se: number;
  lift_tstat: number;
  lift_prob_positive: number;
  lift_prob_negative: number;
}

export interface MonteCarloResult {
  distributions: { [metric: string]: MonteCarloDistribution };
  raw: { [metric: string]: number[] };
  paths: MonteCarloPaths;
  diagnostics: MonteCarloDiagnostics;
  n_sims_requested: number;
  n_sims_completed: number;
  n_sims_failed: number;
  n_sims_full_horizon: number;
  runtime_seconds: number;
  failure_messages?: string[];
}

export async function runMonteCarlo(
  params: MonteCarloParams,
): Promise<MonteCarloResult> {
  const response = await fetch(`${API_BASE}/monte_carlo`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    throw new Error('Monte Carlo simulation failed');
  }
  return response.json();
}
