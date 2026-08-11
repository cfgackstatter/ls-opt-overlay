import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  SimulationParams,
  MonteCarloResult,
  MonteCarloDistribution,
  runMonteCarlo,
} from './api/client';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import './App.css';

const PCT_FIELDS = new Set([
  'mean_return', 'volatility', 'market_vol', 'style_vol',
  'cash_rate', 'margin_rate', 'borrow_fee',
  'long_weight', 'short_weight', 'hard_turnover_limit',
  'max_long_weight', 'max_short_weight',
  'call_otm_pct', 'put_otm_pct', 'call_alpha_barrier', 'put_alpha_barrier',
  'dividend_yield',
]);

const LIFT_COLORS = ['#0f6b5c', '#1f4e79', '#8a4b12', '#3d4a5c', '#0a4f44'];

function App() {
  const [params, setParams] = useState<SimulationParams | null>(null);
  const [showUniverse, setShowUniverse] = useState(false);
  const [showFinancing, setShowFinancing] = useState(false);
  const [showOptions, setShowOptions] = useState(true);
  const [showStrategy, setShowStrategy] = useState(true);
  const [mcTrigger, setMcTrigger] = useState(0);
  const [nSims, setNSims] = useState(50);

  React.useEffect(() => {
    fetch('/defaults')
      .then(r => {
        if (!r.ok) throw new Error(`/defaults returned ${r.status}`);
        return r.json();
      })
      .then((data: SimulationParams) => setParams(data))
      .catch(err => console.error('Failed to load defaults:', err));
  }, []);

  const {
    data: mcData,
    isLoading: isMcLoading,
    error: mcError,
  } = useQuery<MonteCarloResult>({
    queryKey: ['monte_carlo', mcTrigger],
    queryFn: () => runMonteCarlo({ simulation_params: params!, n_sims: nSims }),
    enabled: mcTrigger > 0 && params !== null,
    refetchOnWindowFocus: false,
    staleTime: Infinity,
  });

  if (!params) {
    return <div className="loading-screen">Loading defaults from API…</div>;
  }

  const toDecimal = (key: string, v: number) => (PCT_FIELDS.has(key) ? v * 0.01 : v);

  const handleUniverseChange = (key: string, value: number) => {
    setParams(prev => ({ ...prev!, universe: { ...prev!.universe, [key]: toDecimal(key, value) } }));
  };
  const handleFinancingChange = (key: string, value: number) => {
    setParams(prev => ({ ...prev!, financing: { ...prev!.financing, [key]: toDecimal(key, value) } }));
  };
  const handleStrategyChange = (key: string, value: number) => {
    setParams(prev => ({ ...prev!, strategy: { ...prev!.strategy, [key]: toDecimal(key, value) } }));
  };
  const handleStrategyStringChange = (key: string, value: string) => {
    setParams(prev => ({ ...prev!, strategy: { ...prev!.strategy, [key]: value } }));
  };
  const handleOptionsChange = (key: string, value: number) => {
    setParams(prev => ({ ...prev!, options: { ...prev!.options, [key]: toDecimal(key, value) } }));
  };

  const netExposure =
    (params.strategy.long_weight - params.strategy.short_weight) * 100;
  const cashResidual = 100 - netExposure;

  const liftChartData = mcData
    ? mcData.paths.months.map((m, idx) => {
        const point: Record<string, number> = { month: m };
        mcData.paths.options_lift_paths.forEach((path, j) => {
          point[`sim_${j}`] = path[idx];
        });
        return point;
      })
    : [];

  const LiftTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    const vals = payload
      .map((e: any) => Number(e.value))
      .filter((v: number) => Number.isFinite(v));
    if (!vals.length) return null;
    const mean = vals.reduce((a: number, b: number) => a + b, 0) / vals.length;
    const sorted = [...vals].sort((a, b) => a - b);
    const p25 = sorted[Math.floor(0.25 * (sorted.length - 1))];
    const p75 = sorted[Math.floor(0.75 * (sorted.length - 1))];
    return (
      <div className="recharts-tooltip">
        <div><strong>Month {label}</strong></div>
        <div>Mean lift: {mean.toFixed(2)} pp</div>
        <div>IQR: {p25.toFixed(2)} – {p75.toFixed(2)}</div>
        <div style={{ color: '#6b7a8d' }}>{vals.length} paths</div>
      </div>
    );
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="hero-kicker">Monte Carlo research lab</p>
        <h1 className="hero-title">Sticky Alpha Overlay</h1>
        <p className="hero-lede">
          Test whether selling covered calls and puts on low-conviction holdings
          improves a factor long–short book after transaction costs keep names
          in the portfolio. Equity construction is mean–variance with TC in the
          objective; options are priced at theoretical BSM mid.
        </p>
        <div className="hero-meta">
          <span>
            Book:{' '}
            <strong>
              {(params.strategy.long_weight * 100).toFixed(0)}/
              {(params.strategy.short_weight * 100).toFixed(0)}
            </strong>
            {' '}({netExposure.toFixed(0)}% net, {cashResidual.toFixed(0)}% cash)
          </span>
          <span>
            Overlay:{' '}
            <strong>{params.options.enabled ? 'on' : 'off'}</strong>
          </span>
          <span>
            Horizon:{' '}
            <strong>{params.strategy.strategy_length} mo</strong>
          </span>
        </div>
      </header>

      <CollapsiblePanel
        title="Market DGP"
        blurb="Synthetic multi-factor equity universe that generates prices and the predictive signal."
        open={showUniverse}
        onToggle={() => setShowUniverse(v => !v)}
      >
        <div className="params-grid">
          <Field
            label="Assets"
            value={params.universe.n_assets}
            step={1}
            onChange={v => handleUniverseChange('n_assets', v)}
            tip="Number of names in the simulated universe."
          />
          <Field
            label="Equity premium % ann."
            value={params.universe.mean_return * 100}
            step={0.1}
            onChange={v => handleUniverseChange('mean_return', v)}
            tip="Annualized drift of the price GBM (total return; no dividends)."
          />
          <Field
            label="Name vol % ann."
            value={params.universe.volatility * 100}
            step={0.1}
            onChange={v => handleUniverseChange('volatility', v)}
            tip="Target total annualized volatility per name. Idiosyncratic vol is backed out after market/style betas; names disperse ~0.5×–1.5×."
          />
          <Field
            label="DGP residual IC"
            value={params.universe.ic}
            step={0.01}
            onChange={v => handleUniverseChange('ic', v)}
            tip="True residual information coefficient: corr(signal, idiosyncratic shock) after common factors. This is ground-truth foresight in the simulator."
          />
          <Field
            label="Signal AR(1)"
            value={params.universe.factor_autocorr}
            step={0.05}
            onChange={v => handleUniverseChange('factor_autocorr', v)}
            tip="Persistence of the stock-level alpha signal process."
          />
          <Field
            label="Market factor vol %"
            value={params.universe.market_vol * 100}
            step={0.5}
            onChange={v => handleUniverseChange('market_vol', v)}
            tip="Annualized volatility of the common market factor."
          />
          <Field
            label="Style factor vol %"
            value={params.universe.style_vol * 100}
            step={0.5}
            onChange={v => handleUniverseChange('style_vol', v)}
            tip="Annualized volatility of size/value style factors."
          />
          <Field
            label="Avg market β"
            value={params.universe.avg_beta}
            step={0.05}
            onChange={v => handleUniverseChange('avg_beta', v)}
            tip="Mean market beta across names."
          />
          <Field
            label="Student-t df"
            value={params.universe.student_t_df}
            step={1}
            onChange={v => handleUniverseChange('student_t_df', v)}
            tip="Degrees of freedom for idio shocks. Lower → fatter tails; large → near-Gaussian."
          />
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="Financing"
        blurb="Cash interest, margin, and stock-loan fees on the post-trade book."
        open={showFinancing}
        onToggle={() => setShowFinancing(v => !v)}
      >
        <div className="params-grid">
          <Field
            label="Cash rate % ann."
            value={params.financing.cash_rate * 100}
            step={0.1}
            onChange={v => handleFinancingChange('cash_rate', v)}
            tip="Interest earned on positive cash. Also used as the risk-free rate in BSM option pricing."
          />
          <Field
            label="Margin rate % ann."
            value={params.financing.margin_rate * 100}
            step={0.1}
            onChange={v => handleFinancingChange('margin_rate', v)}
            tip="Interest paid when cash is negative (levered / net-long funding)."
          />
          <Field
            label="Borrow fee % ann."
            value={params.financing.borrow_fee * 100}
            step={0.1}
            onChange={v => handleFinancingChange('borrow_fee', v)}
            tip="Stock-loan fee on short notional."
          />
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="Equity strategy"
        blurb="Mean–variance sleeves with Grinold alpha, risk model, and transaction costs in the objective."
        open={showStrategy}
        onToggle={() => setShowStrategy(v => !v)}
      >
        <div className="params-grid">
          <Field
            label="Horizon months"
            value={params.strategy.strategy_length}
            step={6}
            onChange={v => handleStrategyChange('strategy_length', v)}
            tip="Number of monthly rebalances after the lookback warm-up."
          />
          <Field
            label="Lookback months"
            value={params.strategy.lookback}
            step={6}
            onChange={v => handleStrategyChange('lookback', v)}
            tip="History window used for covariance estimation before trading starts."
          />
          <Field
            label="Risk aversion λ"
            value={params.strategy.risk_aversion}
            step={0.5}
            onChange={v => handleStrategyChange('risk_aversion', v)}
            tip="Penalty on w′Σw in the Markowitz objective. Higher λ → smaller active risk."
          />
          <Field
            label="Long sleeve % NAV"
            value={params.strategy.long_weight * 100}
            step={1}
            onChange={v => handleStrategyChange('long_weight', v)}
            tip="Gross long exposure. Examples: 100 long-only, 130 for 130/30, 150 levered long."
          />
          <Field
            label="Short sleeve % NAV"
            value={params.strategy.short_weight * 100}
            step={1}
            onChange={v => handleStrategyChange('short_weight', v)}
            tip="Gross short exposure. 0 = long-only. Cash residual ≈ 100 − (long − short)."
          />
          <Field
            label="Max long name %"
            value={params.strategy.max_long_weight * 100}
            step={0.5}
            onChange={v => handleStrategyChange('max_long_weight', v)}
            tip="Hard cap on any single long weight."
          />
          <Field
            label="Max short name %"
            value={params.strategy.max_short_weight * 100}
            step={0.5}
            onChange={v => handleStrategyChange('max_short_weight', v)}
            tip="Hard cap on any single short weight (absolute)."
          />
          <Field
            label="Signal IC (Grinold)"
            value={params.strategy.signal_ic}
            step={0.01}
            onChange={v => handleStrategyChange('signal_ic', v)}
            tip="IC used in α = IC · σ · z. Can differ from DGP residual IC to study misspecification."
          />
          <Field
            label="TC κ₁ bps"
            value={params.strategy.transaction_cost_bps}
            step={1}
            onChange={v => handleStrategyChange('transaction_cost_bps', v)}
            tip="Linear transaction cost in the objective and in cash: bps of NAV per unit of Σ|Δw|. Primary control of turnover."
          />
          <Field
            label="Impact κ₂"
            value={params.strategy.market_impact_coef}
            step={0.001}
            onChange={v => handleStrategyChange('market_impact_coef', v)}
            tip="Quadratic market-impact penalty on Σ(Δwᵢ²). Soft; not booked to cash. 0 = off."
          />
          <Field
            label="Hard turnover cap %"
            value={params.strategy.hard_turnover_limit * 100}
            step={1}
            onChange={v => handleStrategyChange('hard_turnover_limit', v)}
            tip="Optional Σ|Δw| ≤ cap. Prefer 0 and let κ₁ set turnover. Initial book build is exempt."
          />
          <Field
            label="EWMA half-life mo"
            value={params.strategy.cov_halflife}
            step={6}
            onChange={v => handleStrategyChange('cov_halflife', v)}
            tip="Half-life for EWMA covariance. Ignored when covariance method is Ledoit–Wolf."
          />
          <SelectField
            label="Alpha method"
            value={params.strategy.alpha_method}
            onChange={v => handleStrategyStringChange('alpha_method', v)}
            tip="How the cross-sectional signal becomes expected return α."
            options={[
              { value: 'grinold', label: 'Grinold IC·σ·z' },
              { value: 'rank_grinold', label: 'Rank Grinold' },
              { value: 'zscore', label: 'Z-score only' },
            ]}
          />
          <SelectField
            label="Covariance"
            value={params.strategy.cov_method}
            onChange={v => handleStrategyStringChange('cov_method', v)}
            tip="Risk model for Σ in the optimizer."
            options={[
              { value: 'ewma', label: 'EWMA' },
              { value: 'ledoit', label: 'Ledoit–Wolf' },
            ]}
          />
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="Options overwrite"
        blurb="Covered calls on weak longs and covered puts on unwanted shorts, after the equity rebalance."
        open={showOptions}
        onToggle={() => setShowOptions(v => !v)}
      >
        <div className="params-grid">
          <label className="field-check">
            <input
              type="checkbox"
              checked={params.options.enabled}
              onChange={e =>
                setParams(prev => ({
                  ...prev!,
                  options: { ...prev!.options, enabled: e.target.checked },
                }))
              }
            />
            Enable overlay
          </label>
          <Field
            label="Call OTM %"
            value={params.options.call_otm_pct * 100}
            step={0.5}
            onChange={v => handleOptionsChange('call_otm_pct', v)}
            tip="Call strike K = S·(1 + otm). European, 1-month tenor, written 1× covered."
          />
          <Field
            label="Put OTM %"
            value={params.options.put_otm_pct * 100}
            step={0.5}
            onChange={v => handleOptionsChange('put_otm_pct', v)}
            tip="Put strike K = S·(1 − otm). Covered puts on short names."
          />
          <Field
            label="Call α barrier %/mo"
            value={params.options.call_alpha_barrier * 100}
            step={0.05}
            onChange={v => handleOptionsChange('call_alpha_barrier', v)}
            tip="Sell calls on longs with monthly α ≤ barrier. 0 ≈ overwrite any non-positive alpha name."
          />
          <Field
            label="Put α barrier %/mo"
            value={params.options.put_alpha_barrier * 100}
            step={0.05}
            onChange={v => handleOptionsChange('put_alpha_barrier', v)}
            tip="Sell puts on shorts with monthly α ≥ barrier. 0 ≈ overwrite any non-negative alpha short."
          />
          <Field
            label="Contract fee $"
            value={params.options.contract_fee}
            step={0.05}
            onChange={v => handleOptionsChange('contract_fee', v)}
            tip="Fee per contract on open and on ITM exercise."
          />
          <Field
            label="Quoted spread bps"
            value={params.options.spread_bps}
            step={5}
            onChange={v => handleOptionsChange('spread_bps', v)}
            tip="Full bid–ask spread. Seller receives BSM mid minus half-spread."
          />
          <Field
            label="Dividend yield q %"
            value={params.options.dividend_yield * 100}
            step={0.1}
            onChange={v => handleOptionsChange('dividend_yield', v)}
            tip="BSM continuous dividend yield. Keep 0 to match the non-dividend equity DGP."
          />
        </div>
      </CollapsiblePanel>

      <div className="panel">
        <div className="run-bar">
          <Field
            label="Monte Carlo paths"
            value={nSims}
            step={10}
            onChange={v => setNSims(Math.max(1, Math.round(v)))}
            tip="Independent market paths. Use tens for a smoke test; hundreds for stable lift t-stats and percentiles."
          />
          <button
            className="run-button"
            onClick={() => setMcTrigger(prev => prev + 1)}
            disabled={isMcLoading}
          >
            {isMcLoading ? 'Running Monte Carlo…' : 'Run Monte Carlo'}
          </button>
        </div>
        {mcError && (
          <div className="error-banner">
            {(mcError as Error).message}
          </div>
        )}
      </div>

      {mcData && (
        <>
          <section className="panel">
            <div className="panel-head panel-head-static">
              <div className="panel-titles">
                <h2 className="panel-title">Overlay validity</h2>
                <p className="panel-blurb">
                  Cross-sectional diagnostics on annualized options lift (overlay − base) across paths.
                </p>
              </div>
            </div>
            <div className="panel-body">
              <ul className="results-meta">
                <li>
                  Completed{' '}
                  <strong>
                    {mcData.n_sims_completed}/{mcData.n_sims_requested}
                  </strong>
                </li>
                <li>
                  Full horizon <strong>{mcData.n_sims_full_horizon}</strong>
                </li>
                {mcData.n_sims_failed > 0 && (
                  <li>
                    Failed <strong>{mcData.n_sims_failed}</strong>
                  </li>
                )}
                <li>
                  Runtime <strong>{mcData.runtime_seconds}s</strong>
                </li>
              </ul>

              <div className="diag-grid">
                <Diag
                  label="Mean lift"
                  value={`${fmt(mcData.diagnostics.lift_mean, 2)}%`}
                  hint="Average ann. return overlay − base"
                  emphasis
                />
                <Diag
                  label="Std. error"
                  value={`${fmt(mcData.diagnostics.lift_se, 2)}%`}
                  hint="σ / √N of lift"
                />
                <Diag
                  label="t-stat"
                  value={fmt(mcData.diagnostics.lift_tstat, 2)}
                  hint="mean / SE (large-sample)"
                  emphasis
                />
                <Diag
                  label="P(lift > 0)"
                  value={`${(mcData.diagnostics.lift_prob_positive * 100).toFixed(0)}%`}
                  hint="Hit rate across completed sims"
                  emphasis
                />
                <Diag
                  label="Lift σ"
                  value={`${fmt(mcData.diagnostics.lift_std, 2)}%`}
                  hint="Cross-sim dispersion of lift"
                />
                <Diag
                  label="P(lift < 0)"
                  value={`${(mcData.diagnostics.lift_prob_negative * 100).toFixed(0)}%`}
                  hint="Share of paths where overlay underperforms"
                />
              </div>

              <h3 className="section-label">Base vs overlay (distribution means)</h3>
              <ComparisonTable mc={mcData} />
            </div>
          </section>

          <section className="panel">
            <div className="panel-head panel-head-static">
              <div className="panel-titles">
                <h2 className="panel-title">Cumulative options lift paths</h2>
                <p className="panel-blurb">
                  Each line is one Monte Carlo path: cumulative return of overlay minus base (percentage points).
                </p>
              </div>
            </div>
            <div className="panel-body">
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={liftChartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#d5dde8" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="month"
                      tick={{ fill: '#6b7a8d', fontSize: 11 }}
                      label={{ value: 'Month', position: 'insideBottom', offset: -2, fill: '#6b7a8d', fontSize: 11 }}
                    />
                    <YAxis
                      tickFormatter={v => `${Math.round(v)}`}
                      tick={{ fill: '#6b7a8d', fontSize: 11 }}
                      label={{ value: 'pp', angle: -90, position: 'insideLeft', fill: '#6b7a8d', fontSize: 11 }}
                    />
                    <Tooltip content={<LiftTooltip />} />
                    {mcData.paths.options_lift_paths.map((_, j) => (
                      <Line
                        key={j}
                        type="linear"
                        dataKey={`sim_${j}`}
                        stroke={LIFT_COLORS[j % LIFT_COLORS.length]}
                        strokeOpacity={0.28}
                        strokeWidth={1.25}
                        dot={false}
                        isAnimationActive={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="chart-note">
                Hover shows the cross-sectional mean and IQR of lift at that month. Short paths (e.g. ruin) are held flat at their terminal lift.
              </p>
            </div>
          </section>
        </>
      )}
    </div>
  );
}

function fmt(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(digits);
}

function distMean(mc: MonteCarloResult, key: string): number | null {
  const d = mc.distributions[key];
  return d ? d.mean : null;
}

function formatCell(d: MonteCarloDistribution | undefined, kind: 'pct' | 'num'): React.ReactNode {
  if (!d) return '—';
  const main = kind === 'pct' ? `${d.mean.toFixed(2)}%` : d.mean.toFixed(2);
  return (
    <>
      {main}
      <span className="metric-sub">
        med {d.median.toFixed(2)} · p25 {d.p25.toFixed(2)} · p75 {d.p75.toFixed(2)}
      </span>
    </>
  );
}

function ComparisonTable({ mc }: { mc: MonteCarloResult }) {
  const rows: { label: string; base: string; ovl: string; kind: 'pct' | 'num'; tip: string }[] = [
    {
      label: 'Ann. return',
      base: 'annualized_return_base',
      ovl: 'annualized_return_with_options',
      kind: 'pct',
      tip: 'Compounded annualized portfolio return.',
    },
    {
      label: 'Ann. alpha vs EW bench',
      base: 'alpha_base',
      ovl: 'alpha_with_options',
      kind: 'pct',
      tip: 'Mean active return vs equal-weight net-exposure benchmark, ×12.',
    },
    {
      label: 'Sharpe',
      base: 'sharpe_base',
      ovl: 'sharpe_with_options',
      kind: 'num',
      tip: 'Mean / vol of monthly returns, annualized √12. Not excess over cash.',
    },
    {
      label: 'Information ratio',
      base: 'information_ratio_base',
      ovl: 'information_ratio_with_options',
      kind: 'num',
      tip: 'Annualized alpha / tracking error vs the benchmark.',
    },
    {
      label: 'Max drawdown',
      base: 'max_drawdown_base',
      ovl: 'max_drawdown_with_options',
      kind: 'pct',
      tip: 'Worst peak-to-trough decline within each path (mean across sims).',
    },
    {
      label: 'Avg turnover',
      base: 'avg_turnover_base',
      ovl: 'avg_turnover_with_options',
      kind: 'pct',
      tip: 'Mean monthly Σ|Δw| × 100 (excludes founding trade).',
    },
    {
      label: 'Options income (ann.)',
      base: '',
      ovl: 'avg_options_income',
      kind: 'pct',
      tip: 'Mean monthly options P&L / NAV, ×12. Premium collected minus settlements/fees.',
    },
  ];

  const bench = distMean(mc, 'annualized_return_benchmark');

  return (
    <table className="compare-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Base</th>
          <th>Overlay</th>
          <th>Δ</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(row => {
          const b = row.base ? mc.distributions[row.base] : undefined;
          const o = mc.distributions[row.ovl];
          const bMean = b?.mean;
          const oMean = o?.mean;
          let delta: React.ReactNode = '—';
          if (oMean != null && bMean != null) {
            const d = oMean - bMean;
            const cls = d > 0 ? 'delta-pos' : d < 0 ? 'delta-neg' : '';
            delta = (
              <span className={cls}>
                {d > 0 ? '+' : ''}
                {row.kind === 'pct' ? `${d.toFixed(2)}%` : d.toFixed(2)}
              </span>
            );
          } else if (oMean != null && !row.base) {
            delta = row.kind === 'pct' ? `${oMean.toFixed(2)}%` : oMean.toFixed(2);
          }
          return (
            <tr key={row.label} title={row.tip}>
              <td>
                {row.label}
                <span className="metric-sub">{row.tip}</span>
              </td>
              <td>{row.base ? formatCell(b, row.kind) : '—'}</td>
              <td>{formatCell(o, row.kind)}</td>
              <td>{delta}</td>
            </tr>
          );
        })}
        <tr title="Equal-weight index scaled to the strategy’s net equity exposure, plus cash residual.">
          <td>
            Benchmark ann. return
            <span className="metric-sub">
              EW net exposure + cash residual
            </span>
          </td>
          <td colSpan={3} style={{ textAlign: 'left' }}>
            {bench == null ? '—' : `${bench.toFixed(2)}% mean`}
            {mc.distributions.annualized_return_benchmark && (
              <span className="metric-sub">
                med {mc.distributions.annualized_return_benchmark.median.toFixed(2)} · p25{' '}
                {mc.distributions.annualized_return_benchmark.p25.toFixed(2)} · p75{' '}
                {mc.distributions.annualized_return_benchmark.p75.toFixed(2)}
              </span>
            )}
          </td>
        </tr>
      </tbody>
    </table>
  );
}

function Diag({
  label,
  value,
  hint,
  emphasis,
}: {
  label: string;
  value: string;
  hint: string;
  emphasis?: boolean;
}) {
  return (
    <div className={`diag-card${emphasis ? ' emphasis' : ''}`}>
      <div className="diag-label">{label}</div>
      <div className="diag-value">{value}</div>
      <div className="diag-hint">{hint}</div>
    </div>
  );
}

function CollapsiblePanel({
  title,
  blurb,
  open,
  onToggle,
  children,
}: {
  title: string;
  blurb: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <button type="button" className="panel-head" onClick={onToggle} aria-expanded={open}>
        <div className="panel-titles">
          <h2 className="panel-title">{title}</h2>
          <p className="panel-blurb">{blurb}</p>
        </div>
        <span className="panel-toggle">{open ? 'Collapse' : 'Expand'}</span>
      </button>
      {open && <div className="panel-body">{children}</div>}
    </section>
  );
}

function Field({
  label,
  value,
  step = 1,
  onChange,
  tip,
}: {
  label: string;
  value: number;
  step?: number;
  onChange: (value: number) => void;
  tip: string;
}) {
  const [showTip, setShowTip] = useState(false);
  const [inputValue, setInputValue] = useState(String(value));

  React.useEffect(() => {
    setInputValue(String(value));
  }, [value]);

  return (
    <div className="field">
      <label className="field-label">
        {label}
        <span
          className="field-help"
          tabIndex={0}
          aria-label={tip}
          onMouseEnter={() => setShowTip(true)}
          onMouseLeave={() => setShowTip(false)}
          onFocus={() => setShowTip(true)}
          onBlur={() => setShowTip(false)}
        >
          ?
        </span>
        {showTip && <div className="field-tooltip" role="tooltip">{tip}</div>}
      </label>
      <input
        className="field-control"
        type="number"
        step={step}
        value={inputValue}
        title={tip}
        onChange={e => {
          const raw = e.target.value;
          setInputValue(raw);
          if (raw === '' || raw === '-' || raw === '+') return;
          const num = Number(raw);
          if (!Number.isNaN(num)) onChange(num);
        }}
      />
    </div>
  );
}

function SelectField({
  label,
  value,
  onChange,
  tip,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  tip: string;
  options: { value: string; label: string }[];
}) {
  const [showTip, setShowTip] = useState(false);
  return (
    <div className="field">
      <label className="field-label">
        {label}
        <span
          className="field-help"
          tabIndex={0}
          aria-label={tip}
          onMouseEnter={() => setShowTip(true)}
          onMouseLeave={() => setShowTip(false)}
          onFocus={() => setShowTip(true)}
          onBlur={() => setShowTip(false)}
        >
          ?
        </span>
        {showTip && <div className="field-tooltip" role="tooltip">{tip}</div>}
      </label>
      <select
        className="field-control"
        value={value}
        title={tip}
        onChange={e => onChange(e.target.value)}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export default App;
