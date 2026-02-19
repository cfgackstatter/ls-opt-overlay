# Long-Short Strategy Simulator

A Monte Carlo simulator for a factor-based long-short equity strategy with an options overlay, built with a Python backend and React frontend.

## Features

- Factor-based portfolio optimization with turnover constraints and position limits
- Options overlay (covered calls on longs, cash-secured puts on shorts) with configurable strike and cost parameters
- Monte Carlo simulation across many random universes for robust strategy evaluation
- Distribution-based performance metrics: annualized return, Sharpe, information ratio, max drawdown
- Options lift fan chart showing cumulative overlay benefit across all simulation paths

## Project Structure

```text
├── backend/
│   ├── models.py          # Pydantic data models
│   ├── simulator.py       # Core simulation engine
│   ├── strategy.py        # Portfolio construction and optimization
│   ├── options.py         # Options overlay logic
│   ├── alpha.py           # Factor score generation
│   ├── risk.py            # Risk calculations
│   ├── optimizer.py       # Portfolio optimizer
│   ├── monte_carlo.py     # Monte Carlo engine
│   └── api.py             # FastAPI endpoints
├── frontend/
│   └── src/
│       ├── App.tsx         # Main React app
│       ├── App.css         # Styles
│       └── api/client.ts   # API client
├── notebook-charts.ipynb   # Charting notebook
├── notebook-logging.ipynb  # Logging notebook
└── run.py                  # Backend server launcher
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

## Running

### Start the backend:

```bash
python run.py
```

### Start the frontend in a separate terminal:

```bash
cd frontend
npm run dev
```

The frontend runs on `http://localhost:3000` and expects the backend at `http://localhost:8000`.