const API_BASE = 'http://localhost:8000';

export interface UniverseParams {
  n_assets: number;
  mean_return: number;
  volatility: number;
  ic: number;
  factor_autocorr: number;
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
  turnover_limit: number;
  max_weight: number;
  lookback: number;
  transaction_cost_bps: number;
}

export interface OptionsParams {
  enabled: boolean;
  call_otm_pct: number;
  put_otm_pct: number;
  call_alpha_barrier: number;
  put_alpha_barrier: number;
  contract_fee: number;
  spread_bps: number;
}

export interface SimulationParams {
  universe: UniverseParams;
  financing: FinancingParams;
  strategy: StrategyParams;
  options: OptionsParams;
}

// StrategyResult now only describes one portfolio stream
export interface StrategyResult {
  portfolio_returns: number[];   // monthly %
  financing_costs: number[];     // monthly %
  portfolio_weights: number[][];
  turnovers: number[];           // monthly %
  cumulative_return: number;     // %
  annualized_return: number;     // %
  sharpe_ratio: number;
  avg_turnover: number;          // %
  avg_financing_cost: number;    // %
  options_income: number[];      // monthly %
  avg_options_income: number;    // %
}

// Benchmark leg
export interface BenchmarkResult {
  returns: number[];           // monthly %
  cumulative_return: number;   // %
  annualized_return: number;   // %
  sharpe_ratio: number;
}

// Full simulation result (single path)
export interface SimulationResult {
  base: StrategyResult;
  with_options: StrategyResult;
  benchmark: BenchmarkResult;

  alpha_base: number;                // %
  alpha_with_options: number;        // %
  information_ratio_base: number;
  information_ratio_with_options: number;
  options_lift: number;              // %
}

export async function runSimulation(params: SimulationParams): Promise<SimulationResult> {
  const response = await fetch(`${API_BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    throw new Error('Simulation failed');
  }

  return response.json();
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

export interface MonteCarloResult {
  distributions: { [metric: string]: MonteCarloDistribution };
  raw: { [metric: string]: number[] };
  paths: MonteCarloPaths;
  n_sims_completed: number;
  runtime_seconds: number;
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