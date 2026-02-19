import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  SimulationParams,
  MonteCarloResult,
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

const defaultParams: SimulationParams = {
  universe: {
    n_assets: 50,
    mean_return: 6.0,
    volatility: 20.0,
    ic: 0.1,
    factor_autocorr: 0.8,
  },
  financing: {
    cash_rate: 3.0,
    margin_rate: 5.0,
    borrow_fee: 1.0,
  },
  strategy: {
    strategy_length: 60,
    risk_aversion: 4.0,
    long_weight: 100.0,
    short_weight: 100.0,
    turnover_limit: 15.0,
    max_weight: 10.0,
    lookback: 36,
    transaction_cost_bps: 5.0,
  },
  options: {
    enabled: true, // backend always runs base + overlay; this just passes the overlay params
    call_otm_pct: 0.0,
    put_otm_pct: 0.0,
    call_alpha_barrier: -0.1,
    put_alpha_barrier: 0.1,
    contract_fee: 0.65,
    spread_bps: 100.0,
  },
};

function App() {
  const [params, setParams] = useState<SimulationParams>(defaultParams);
  const [showUniverse, setShowUniverse] = useState(false);
  const [showFinancing, setShowFinancing] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [nSims, setNSims] = useState(10);
  const [mcTrigger, setMcTrigger] = useState(0);

  const {
    data: mcData,
    isLoading: isMcLoading,
    error: mcError,
  } = useQuery<MonteCarloResult>({
    queryKey: ['monte_carlo', mcTrigger],
    queryFn: () =>
      runMonteCarlo({
        simulation_params: params,
        n_sims: nSims,
      }),
    enabled: mcTrigger > 0,
    refetchOnWindowFocus: false,
    staleTime: Infinity,
  });

  const handleUniverseChange = (key: string, value: number) => {
    setParams(prev => ({
      ...prev,
      universe: { ...prev.universe, [key]: value },
    }));
  };

  const handleFinancingChange = (key: string, value: number) => {
    setParams(prev => ({
      ...prev,
      financing: { ...prev.financing, [key]: value },
    }));
  };

  const handleStrategyChange = (key: string, value: number) => {
    setParams(prev => ({
      ...prev,
      strategy: { ...prev.strategy, [key]: value },
    }));
  };

  const handleOptionsChange = (key: string, value: number) => {
    setParams(prev => ({
      ...prev,
      options: { ...prev.options, [key]: value },
    }));
  };

  // Strategy performance chart data: base vs overlay vs benchmark
  const liftChartData = mcData
    ? mcData.paths.months.map((m, idx) => {
        const point: any = { month: m };
        mcData.paths.options_lift_paths.forEach((path, j) => {
          point[`Sim ${j + 1}`] = path[idx];
        });
        return point;
      })
    : [];

  const LiftTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    return (
      <div className="custom-tooltip">
        <div>
          <strong>Month {label}</strong>
        </div>
        {payload.map((entry: any) => (
          <div key={entry.dataKey} style={{ color: entry.color }}>
            {entry.name}: {entry.value.toFixed(2)}%
          </div>
        ))}
      </div>
    );
  };

  const axisPercentFormatter = (value: number): string =>
    `${Math.round(value)}%`;

  const formatPercentDist = (key: string): { main: string; detail: string } => {
    if (!mcData) return { main: '-', detail: '' };
    const d = mcData.distributions[key];
    if (!d) return { main: '-', detail: '' };

    const main = `${d.median.toFixed(2)}%`;
    const detail = `mean ${d.mean.toFixed(2)}%, p25 ${d.p25.toFixed(
      2,
    )}%, p75 ${d.p75.toFixed(2)}%`;

    return { main, detail };
  };

  const formatPlainDist = (key: string): { main: string; detail: string } => {
    if (!mcData) return { main: '-', detail: '' };
    const d = mcData.distributions[key];
    if (!d) return { main: '-', detail: '' };

    const main = d.median.toFixed(2);
    const detail = `mean ${d.mean.toFixed(2)}, p25 ${d.p25.toFixed(
      2,
    )}, p75 ${d.p75.toFixed(2)}`;

    return { main, detail };
  };

  return (
    <div className="app-container">
      <div className="app-content">
        <div className="card">
          <h1 className="header-title">Long-Short Strategy Simulator</h1>
          <p className="header-subtitle">
            Factor-based portfolio optimization with turnover constraints and
            options overlay.
          </p>
        </div>

        {/* Universe Parameters */}
        <div className="card">
          <div
            onClick={() => setShowUniverse(!showUniverse)}
            style={{
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              userSelect: 'none',
            }}
          >
            <h2 className="section-title">Universe Parameters</h2>
            <span>{showUniverse ? '−' : '+'}</span>
          </div>
          {showUniverse && (
            <div className="params-grid">
              <ParamInputWithTooltip
                label="Number of Assets"
                value={params.universe.n_assets}
                step={1}
                onChange={v => handleUniverseChange('n_assets', v)}
                tooltip="Total number of stocks in the simulated universe"
              />
              <ParamInputWithTooltip
                label="Mean Return (%)"
                value={params.universe.mean_return}
                step={0.5}
                onChange={v => handleUniverseChange('mean_return', v)}
                tooltip="Expected annualized return for the equal-weight index"
              />
              <ParamInputWithTooltip
                label="Volatility (%)"
                value={params.universe.volatility}
                step={1}
                onChange={v => handleUniverseChange('volatility', v)}
                tooltip="Target annualized volatility. Individual assets have volatilities from 0.5× to 1.5× this value."
              />
              <ParamInputWithTooltip
                label="IC"
                value={params.universe.ic}
                step={0.01}
                onChange={v => handleUniverseChange('ic', v)}
                tooltip="Correlation between factor scores and realized returns."
              />
              <ParamInputWithTooltip
                label="Factor Autocorr"
                value={params.universe.factor_autocorr}
                step={0.05}
                onChange={v => handleUniverseChange('factor_autocorr', v)}
                tooltip="Persistence of factor scores over time."
              />
            </div>
          )}
        </div>

        {/* Financing Parameters */}
        <div className="card">
          <div
            onClick={() => setShowFinancing(!showFinancing)}
            style={{
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              userSelect: 'none',
            }}
          >
            <h2 className="section-title">Financing Parameters</h2>
            <span>{showFinancing ? '−' : '+'}</span>
          </div>
          {showFinancing && (
            <div className="params-grid">
              <ParamInputWithTooltip
                label="Cash Rate (%)"
                value={params.financing.cash_rate}
                step={0.25}
                onChange={v => handleFinancingChange('cash_rate', v)}
                tooltip="Annual interest rate earned on positive cash balances"
              />
              <ParamInputWithTooltip
                label="Margin Rate (%)"
                value={params.financing.margin_rate}
                step={0.25}
                onChange={v => handleFinancingChange('margin_rate', v)}
                tooltip="Annual interest rate paid on negative cash balances"
              />
              <ParamInputWithTooltip
                label="Borrow Fee (%)"
                value={params.financing.borrow_fee}
                step={0.1}
                onChange={v => handleFinancingChange('borrow_fee', v)}
                tooltip="Annual fee paid to borrow shares for short positions"
              />
            </div>
          )}
        </div>

        {/* Options Overlay Parameters */}
        <div className="card">
          <div
            onClick={() => setShowOptions(!showOptions)}
            style={{
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              userSelect: 'none',
            }}
          >
            <h2 className="section-title">Options Overlay Parameters</h2>
            <span>{showOptions ? '−' : '+'}</span>
          </div>
          {showOptions && (
            <div className="params-grid">
              <ParamInputWithTooltip
                label="Call OTM (%)"
                value={params.options.call_otm_pct}
                step={1}
                onChange={v => handleOptionsChange('call_otm_pct', v)}
                tooltip="Percentage above spot for call strike."
              />
              <ParamInputWithTooltip
                label="Put OTM (%)"
                value={params.options.put_otm_pct}
                step={1}
                onChange={v => handleOptionsChange('put_otm_pct', v)}
                tooltip="Percentage below spot for put strike."
              />
              <ParamInputWithTooltip
                label="Call Alpha Barrier (%)"
                value={params.options.call_alpha_barrier * 100}
                step={0.1}
                onChange={v => handleOptionsChange('call_alpha_barrier', v / 100)}
                tooltip="Monthly alpha threshold in percent for selling calls on long positions."
              />
              <ParamInputWithTooltip
                label="Put Alpha Barrier (%)"
                value={params.options.put_alpha_barrier * 100}
                step={0.1}
                onChange={v => handleOptionsChange('put_alpha_barrier', v / 100)}
                tooltip="Monthly alpha threshold in percent for selling puts on short positions."
              />
              <ParamInputWithTooltip
                label="Contract Fee ($)"
                value={params.options.contract_fee}
                step={0.05}
                onChange={v => handleOptionsChange('contract_fee', v)}
                tooltip="Fixed commission per options contract."
              />
              <ParamInputWithTooltip
                label="Spread (bps)"
                value={params.options.spread_bps}
                step={10}
                onChange={v => handleOptionsChange('spread_bps', v)}
                tooltip="Bid-ask spread cost in basis points."
              />
            </div>
          )}
        </div>

        {/* Strategy Parameters */}
        <div className="card">
          <h2 className="section-title">Strategy Parameters</h2>
          <div className="params-grid">
            <ParamInputWithTooltip
              label="Backtest Length (months)"
              value={params.strategy.strategy_length}
              step={6}
              onChange={v => handleStrategyChange('strategy_length', v)}
              tooltip="Number of months to run the backtest."
            />
            <ParamInputWithTooltip
              label="Risk Aversion"
              value={params.strategy.risk_aversion}
              step={0.5}
              onChange={v => handleStrategyChange('risk_aversion', v)}
              tooltip="Higher values reduce risk-taking."
            />
            <ParamInputWithTooltip
              label="Long Weight (%)"
              value={params.strategy.long_weight}
              step={5}
              onChange={v => handleStrategyChange('long_weight', v)}
              tooltip="Gross long exposure."
            />
            <ParamInputWithTooltip
              label="Short Weight (%)"
              value={params.strategy.short_weight}
              step={5}
              onChange={v => handleStrategyChange('short_weight', v)}
              tooltip="Gross short exposure."
            />
            <ParamInputWithTooltip
              label="Turnover Limit (%)"
              value={params.strategy.turnover_limit}
              step={5}
              onChange={v => handleStrategyChange('turnover_limit', v)}
              tooltip="Max one-way turnover per rebalance."
            />
            <ParamInputWithTooltip
              label="Max Position (%)"
              value={params.strategy.max_weight}
              step={1}
              onChange={v => handleStrategyChange('max_weight', v)}
              tooltip="Max absolute position size for any stock."
            />
            <ParamInputWithTooltip
              label="Lookback (months)"
              value={params.strategy.lookback}
              step={6}
              onChange={v => handleStrategyChange('lookback', v)}
              tooltip="History used to estimate covariance."
            />
            <ParamInputWithTooltip
              label="Transaction Cost (bps)"
              value={params.strategy.transaction_cost_bps}
              step={1}
              onChange={v => handleStrategyChange('transaction_cost_bps', v)}
              tooltip="Cost applied to traded notional."
            />
            <ParamInputWithTooltip
              label="Number of simulations"
              value={nSims}
              step={1}
              onChange={v => setNSims(Math.max(1, Math.round(v)))}
              tooltip="Number of Monte Carlo runs. Higher = smoother distributions, slower runtime."
            />
          </div>

          <div className="run-button-wrapper">
            <button
              className="run-button"
              onClick={() => setMcTrigger(prev => prev + 1)}
              disabled={isMcLoading}
              style={{ width: '100%' }}
            >
              {isMcLoading ? 'Running...' : 'Run'}
            </button>
          </div>

          {mcError && (
            <div className="error-message">
              Error: {(mcError as Error).message}
            </div>
          )}
        </div>

        {mcData && (
          <>
            <div className="card">
              <h2 className="section-title">Strategy Performance (Monte Carlo)</h2>
              <div className="stats-grid">
                {/* Annualized returns (percent) */}
                {(() => {
                  const d = formatPercentDist('annualized_return_base');
                  return (
                    <StatBox
                      label="Ann. Return (w/o Options)"
                      value={d.main}
                      detail={d.detail}
                    />
                  );
                })()}
                {(() => {
                  const d = formatPercentDist('annualized_return_with_options');
                  return (
                    <StatBox
                      label="Ann. Return (w/ Options)"
                      value={d.main}
                      detail={d.detail}
                    />
                  );
                })()}
                {(() => {
                  const d = formatPercentDist('annualized_return_benchmark');
                  return (
                    <StatBox
                      label="Ann. Return (Benchmark)"
                      value={d.main}
                      detail={d.detail}
                    />
                  );
                })()}

                {/* Options lift (percent) */}
                {(() => {
                  const d = formatPercentDist('options_lift');
                  return (
                    <StatBox
                      label="Options Lift"
                      value={d.main}
                      detail={d.detail}
                    />
                  );
                })()}

                {/* Sharpe ratios (plain) */}
                {(() => {
                  const d = formatPlainDist('sharpe_base');
                  return (
                    <StatBox
                      label="Sharpe (w/o Options)"
                      value={d.main}
                      detail={d.detail}
                    />
                  );
                })()}
                {(() => {
                  const d = formatPlainDist('sharpe_with_options');
                  return (
                    <StatBox
                      label="Sharpe (w/ Options)"
                      value={d.main}
                      detail={d.detail}
                    />
                  );
                })()}

                {/* Information ratios (plain) */}
                {(() => {
                  const d = formatPlainDist('information_ratio_base');
                  return (
                    <StatBox
                      label="Information Ratio (w/o Options)"
                      value={d.main}
                      detail={d.detail}
                    />
                  );
                })()}
                {(() => {
                  const d = formatPlainDist('information_ratio_with_options');
                  return (
                    <StatBox
                      label="Information Ratio (w/ Options)"
                      value={d.main}
                      detail={d.detail}
                    />
                  );
                })()}

                {/* Max drawdown (percent) */}
                {(() => {
                  const d = formatPercentDist('max_drawdown_base');
                  return (
                    <StatBox
                      label="Max Drawdown (w/o Options)"
                      value={d.main}
                      detail={d.detail}
                    />
                  );
                })()}
                {(() => {
                  const d = formatPercentDist('max_drawdown_with_options');
                  return (
                    <StatBox
                      label="Max Drawdown (w/ Options)"
                      value={d.main}
                      detail={d.detail}
                    />
                  );
                })()}
              </div>
            </div>

            {/* Options lift fan chart stays below */}
            <div className="card">
              <h2 className="section-title">Options Lift Over Time (Monte Carlo)</h2>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={liftChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis tickFormatter={axisPercentFormatter} />
                    <Tooltip content={<LiftTooltip />} />
                    {mcData.paths.options_lift_paths.map((_, j) => (
                      <Line
                        key={j}
                        type="linear"
                        dataKey={`Sim ${j + 1}`}
                        stroke="#10b981"
                        strokeOpacity={0.2}
                        dot={false}
                        isAnimationActive={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

interface ParamInputWithTooltipProps {
  label: string;
  value: number;
  step?: number;
  onChange: (value: number) => void;
  tooltip: string;
}

const ParamInputWithTooltip: React.FC<ParamInputWithTooltipProps> = ({
  label,
  value,
  step = 1,
  onChange,
  tooltip,
}) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const [inputValue, setInputValue] = useState(String(value));

  React.useEffect(() => {
    setInputValue(String(value));
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setInputValue(raw);

    if (raw === '' || raw === '-' || raw === '+') {
      return;
    }

    const num = Number(raw);
    if (!Number.isNaN(num)) {
      onChange(num);
    }
  };

  return (
    <div className="param-input">
      <label className="param-label">
        {label}
        <span
          className="info-icon"
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
        >
          i
        </span>
        {showTooltip && <div className="tooltip">{tooltip}</div>}
      </label>
      <input
        type="number"
        step={step}
        value={inputValue}
        onChange={handleChange}
        className="param-field"
      />
    </div>
  );
};

interface StatBoxProps {
  label: string;
  value: string;      // main value (median)
  detail?: string;    // secondary line (mean, p25, p75)
  target?: string;
}

const StatBox: React.FC<StatBoxProps> = ({ label, value, detail, target }) => (
  <div className="stat-box">
    <div className="stat-label">{label}</div>
    <div className="stat-value">{value}</div>
    {detail && <div className="stat-target">{detail}</div>}
    {target && <div className="stat-target">Target: {target}</div>}
  </div>
);

export default App;
