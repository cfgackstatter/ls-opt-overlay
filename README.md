# Long-Short Strategy Simulator

Monte Carlo lab for a factor-based long–short equity book with an optional
**sticky-alpha options overwrite**. The question the project answers:

> After transaction costs keep names in the book even when alpha is no longer
> attractive, does selling covered calls (on weak longs) and covered puts (on
> unwanted shorts) improve risk-adjusted performance?

The stack is a FastAPI + NumPy/CVXPY backend and a React frontend. Defaults live
in `backend/models.py` and are served to the UI via `/defaults`.

---

## Architecture

```text
Market DGP  →  Alpha / Risk  →  Markowitz + TC  →  Cash / financing
                                      ↓
                         Sticky-alpha options overlay (optional)
                                      ↓
                    Monte Carlo across independent price paths
```

| Layer | Module | Role |
|-------|--------|------|
| Market | `backend/market.py` | Multi-factor monthly GBM with residual IC signal |
| Alpha | `backend/alpha.py` | Grinold `α = IC · σ · z` (or rank / raw z) |
| Risk | `backend/risk.py` | EWMA or Ledoit–Wolf covariance |
| Optimizer | `backend/optimizer.py` | Mean–variance + linear/quadratic TC (DPP/OSQP) |
| Strategy | `backend/strategy.py` | Monthly rebalance, cash identity, financing, options |
| Options | `backend/options.py` | BSM mid pricing + covered overwrite rules |
| Simulator | `backend/simulator.py` | One path: base vs overlay vs benchmark |
| Monte Carlo | `backend/monte_carlo.py` | Parallel independent seeds + diagnostics |
| API | `backend/api.py` | FastAPI endpoints |
| UI | `frontend/` | Parameter panel + MC distributions / lift paths |

---

## Market data-generating process

Synthetic equities (no dividends by default):

- Common factors: market + size + value, with betas and idiosyncratic vol backed
  out from a target total vol
- Cross-sectional residual IC (no look-ahead) drives a predictive signal
- Log returns / GBM prices, Student-t shocks, shared stochastic log-vol

The signal the portfolio *sees* is separate from the DGP IC via
`strategy.signal_ic`, so you can study misspecified or weaker foresight.

---

## Equity strategy

### Sleeves

| Mode | `long_weight` | `short_weight` |
|------|---------------|----------------|
| Long-only | 1.0 | 0.0 |
| 130/30 | 1.3 | 0.3 |
| Market-neutral | 1.0 | 1.0 |
| Levered long | 1.5 | 0.0 |

Cash residual after the equity sleeves is `1 − (long − short)`. Financing uses
post-trade cash and short notional (interest on cash/margin, borrow fee).

### Optimization (industry-style)

```text
max_w  α′w − λ w′Σw − κ₁‖Δw‖₁ − κ₂‖Δw‖₂²
```

subject to sleeve sums and per-name long/short caps.

- **κ₁** = `transaction_cost_bps / 1e4` — also booked in cash P&L
- **κ₂** = `market_impact_coef` — soft quadratic impact (not booked to cash)
- **Hard turnover cap** (`hard_turnover_limit`) optional; **off by default**.
  Costs discipline turnover. The initial book build is exempt from a hard cap.

### Accounting

After each rebalance:

```text
stock = w · NAV
cash  = NAV − Σ stock − linear TC  (+ option premium when overlay is on)
```

Options settle at month-end; financing accrues on the post-trade book.

---

## Options overlay

**Rule (sticky alpha overwrite)**

- Long and `α ≤ call_alpha_barrier` → sell 1M OTM **covered calls**
- Short and `α ≥ put_alpha_barrier` → sell 1M OTM **covered puts**

Lot-rounded coverage: `floor(|shares| / 100)` contracts so the equity position
covers the short option.

**Pricing (theoretical backtest)**

- European **Black–Scholes–Merton** mid with risk-model vol, cash rate `r`,
  dividend yield `q` (keep `q = 0` to match the non-dividend GBM)
- Trade the **bid**: mid × (1 − ½·spread)
- **Cash-settle at intrinsic** at the next rebalance (fee only if ITM)

No implied-vol risk premium is assumed. Unconditional short options are ~fair
before costs; any edge must come from **conditioning on alpha**.

---

## Monte Carlo framework

Each simulation:

1. Draws an independent market path via `SeedSequence` (not just `seed + i`)
2. Runs **base** (options off) and **overlay** on that **same** path with
   **separate** optimizers (avoids OSQP warm-start cross-contamination)
3. Records returns, Sharpe, drawdown, turnover, alpha/IR vs an equal-weight
   net-exposure benchmark, and the cumulative options-lift path

Workers run in a `ProcessPoolExecutor`. Failures are counted and surfaced
(`n_sims_failed`, `failure_messages`) instead of silently shrinking the sample.
Short paths (e.g. ruin) are padded by holding the terminal lift for the fan chart.

**Validity diagnostics** (on `options_lift` across completed sims):

| Field | Meaning |
|-------|---------|
| `lift_mean` / `lift_se` | Mean lift and standard error |
| `lift_tstat` | `mean / se` (large-sample) |
| `lift_prob_positive` | Fraction of paths with lift > 0 |
| `n_sims_full_horizon` | Sims that ran the full strategy length |

Use enough sims for stable tails (tens → hundreds). Distributions use **sample**
standard deviation (`ddof=1`). The API runs MC off the event loop so `/health`
stays responsive during long jobs.

---

## Project layout

```text
├── backend/                 # Python package (API + simulation)
│   ├── __init__.py
│   ├── models.py            # Pydantic params / results (defaults source of truth)
│   ├── market.py            # Multi-factor GBM market generator
│   ├── alpha.py             # Alpha construction
│   ├── risk.py              # Covariance estimators
│   ├── optimizer.py         # CVXPY mean–variance + TC
│   ├── strategy.py          # Rebalance loop, cash, financing, options hooks
│   ├── options.py           # BSM pricing + overwrite overlay
│   ├── simulator.py         # Single-path base / overlay / benchmark
│   ├── monte_carlo.py       # Parallel MC + diagnostics
│   └── api.py               # FastAPI app
├── frontend/                # React + Vite UI (:3000, proxies API)
├── notebook-backtest.ipynb  # Single-path stats + charts
├── Makefile                 # install / backend / frontend helpers
├── run.py                   # Thin uvicorn entry for the API
├── requirements.txt
└── README.md
```

### Notebook

`notebook-backtest.ipynb` runs one simulated path and shows summary stats plus
cumulative returns, drawdowns, options lift / income, and turnover. For
multi-path validity (lift t-stat, P(lift>0)), use the UI Monte Carlo button.
Set `WRITE_DETAIL_LOG = True` in the notebook if you need a period-by-period
text log for debugging.

---

## Setup

**Prerequisites:** Python 3.10+, Node.js 18+, Make (optional but handy)

```bash
make install
# equivalent:
#   python3 -m venv venv && source venv/bin/activate
#   pip install -r requirements.txt
#   cd frontend && npm install
```

## Launch (frontend + backend)

You need **two terminals** (API and Vite are separate processes):

```bash
# terminal 1 — API  http://127.0.0.1:8000
make backend
# or:  source venv/bin/activate && python run.py

# terminal 2 — UI   http://localhost:3000  (proxies /health, /defaults, … to :8000)
make frontend
# or:  cd frontend && npm run dev
```

Then open **http://localhost:3000**. Set sleeves / TC / overlay barriers, choose
the number of simulations, and click **Run**.

`run.py` is a thin, standard FastAPI entry (`uvicorn` with reload). The Makefile
is optional sugar — common for polyglot repos so you do not memorize npm/pip
commands.

### Useful API routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/defaults` | Default `SimulationParams` |
| POST | `/simulate` | Single path |
| POST | `/monte_carlo` | Parallel MC (`n_sims`, `base_seed`, …) |

`notebook-backtest.ipynb` calls the Python modules directly for a single-path
analysis without the UI (`make notebook` to open it).
