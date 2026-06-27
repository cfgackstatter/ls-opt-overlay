import numpy as np
import logging
from typing import Optional
from datetime import datetime
from backend.alpha import calculate_alphas
from backend.risk import estimate_covariance_ewma
from backend.optimizer import optimize_portfolio
from backend.options import sell_options_overlay, settle_options

MONTHS_PER_YEAR = 12
INITIAL_CAPITAL = 1_000_000.0


def setup_logger(verbose: bool = False, log_file: Optional[str] = None) -> Optional[logging.Logger]:
    """Setup file logger for strategy execution. Returns None when verbose=False."""
    if not verbose:
        return None
    if log_file is None:
        log_file = f"strategy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = logging.getLogger("strategy")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def calculate_financing_cost(
    cash_dollars: float,
    long_dollars: float,
    cash_rate: float,
    margin_rate: float,
    borrow_fee: float,
    short_dollars: float,
) -> float:
    """Calculate monthly financing cost/income based on dollar positions."""
    cash_income = cash_dollars * (cash_rate / MONTHS_PER_YEAR) if cash_dollars >= 0 else 0.0
    margin_cost = abs(cash_dollars) * (margin_rate / MONTHS_PER_YEAR) if cash_dollars < 0 else 0.0
    borrow_cost = short_dollars * (borrow_fee / MONTHS_PER_YEAR)
    return cash_income - margin_cost - borrow_cost


def clean_small_weights(weights: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """
    Zero out tiny weights and rescale longs/shorts to preserve total exposure.
    """
    cleaned = weights.copy()
    orig_long: float = float(cleaned[cleaned > 0].sum())
    orig_short: float = float(cleaned[cleaned < 0].sum())

    cleaned[np.abs(cleaned) < threshold] = 0.0

    # Rescale longs to maintain total long exposure
    new_long: float = float(cleaned[cleaned > 0].sum())
    if new_long > 0:
        cleaned[cleaned > 0] *= orig_long / new_long

    # Rescale shorts to maintain total short exposure
    new_short: float = float(cleaned[cleaned < 0].sum())
    if new_short < 0:
        cleaned[cleaned < 0] *= orig_short / new_short

    return cleaned


def run_strategy(
    returns: np.ndarray,
    prices: np.ndarray,
    factor_scores: np.ndarray,
    strategy_params: dict[str, float],
    financing_params: dict[str, float],
    options_params: dict,
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> dict:
    """
    Run the long-short strategy over a window of returns and prices.

    At each step:
    - optimize and trade at prices[t-1]
    - hold positions as prices move to prices[t]
    - settle options, apply financing, and record P&L
    """
    logger = setup_logger(verbose, log_file)
    # Single lazy log callable — avoids 30+ `if logger:` guards throughout the loop
    log = logger.info if logger else lambda *a, **k: None

    # --- Parameter extraction ---
    lookback: int = int(strategy_params["lookback"])
    strategy_length: int = int(strategy_params["strategy_length"])
    risk_aversion: float = strategy_params["risk_aversion"]
    target_long: float = strategy_params["long_weight"]
    target_short: float = strategy_params["short_weight"]
    turnover_limit: float = strategy_params["turnover_limit"]
    max_weight: float = strategy_params["max_weight"]
    tx_cost_bps: float = strategy_params["transaction_cost_bps"]

    cash_rate: float = financing_params["cash_rate"]
    margin_rate: float = financing_params["margin_rate"]
    borrow_fee: float = financing_params["borrow_fee"]

    options_enabled: bool = options_params.get("enabled", False)
    call_otm_pct: float = options_params.get("call_otm_pct", 5.0)
    put_otm_pct: float = options_params.get("put_otm_pct", 5.0)
    call_alpha_barrier: float = options_params["call_alpha_barrier"]
    put_alpha_barrier: float = options_params["put_alpha_barrier"]
    contract_fee: float = options_params.get("contract_fee", 0.65)
    spread_bps: float = options_params.get("spread_bps", 10.0)

    n_periods, n_assets = returns.shape

    # --- Tracking lists ---
    portfolio_returns: list[float] = []
    financing_costs: list[float] = []
    portfolio_weights_history: list[list[float]] = []
    turnovers: list[float] = []
    options_income: list[float] = []

    # --- Initial state ---
    cash_dollars: float = INITIAL_CAPITAL
    shares: np.ndarray = np.zeros(n_assets)
    active_options: list[dict] = []

    log("=" * 80)
    log("STRATEGY EXECUTION LOG")
    log("=" * 80)
    log(f"\nInitial Capital: ${INITIAL_CAPITAL:,.2f}")
    log(f"Strategy Parameters: {strategy_params}")
    log(f"Financing Parameters: {financing_params}")
    log(f"Options Parameters: {options_params}")
    log("=" * 80 + "\n")

    for t in range(lookback + 1, lookback + strategy_length + 1):
        if t >= n_periods + 1:
            break

        period_num: int = t - lookback
        log("\n" + "=" * 80)
        log(f"PERIOD {period_num}: [{t-1} -> {t}]")
        log("=" * 80)

        # --- Start of period: value positions at t-1 ---
        stock_values: np.ndarray = shares * prices[t - 1]
        portfolio_value_start: float = float(stock_values.sum() + cash_dollars)
        current_weights: np.ndarray = (
            stock_values / portfolio_value_start if portfolio_value_start > 0 else np.zeros(n_assets)
        )

        log(f"\n--- START OF PERIOD (t={t-1}) ---")
        log(f"Portfolio Value: ${portfolio_value_start:,.2f}")
        log(f"Cash: ${cash_dollars:,.2f}")
        log(f"Stock Positions: ${float(stock_values.sum()):,.2f}")
        log(f"Long Positions: ${float(stock_values[stock_values > 0].sum()):,.2f}")
        log(f"Short Positions: ${float(stock_values[stock_values < 0].sum()):,.2f}")
        log(f"Current Weights (top 5 abs): {sorted(enumerate(current_weights), key=lambda x: abs(x[1]), reverse=True)[:5]}")

        # Estimate covariance from returns window ending at t-1
        cov_matrix: np.ndarray = estimate_covariance_ewma(returns[t - 1 - lookback : t - 1])
        alphas: np.ndarray = calculate_alphas(factor_scores[t - 1], cov_matrix)

        log(f"\nAlphas (top 5): {sorted(enumerate(alphas), key=lambda x: x[1], reverse=True)[:5]}")
        log(f"Alphas (bottom 5): {sorted(enumerate(alphas), key=lambda x: x[1])[:5]}")

        # Optimize: first period has no turnover constraint — pass None to unified function
        is_first: bool = t == lookback + 1
        result: dict = optimize_portfolio(
            alphas, cov_matrix, risk_aversion, target_long, target_short, max_weight,
            current_weights=None if is_first else current_weights,
            turnover_limit=None if is_first else turnover_limit,
        )
        target_weights: np.ndarray = clean_small_weights(result["weights"])
        weight_turnover: float = float(np.abs(target_weights).sum()) if is_first else result["turnover"]

        log(f"\nOptimization: {'SIMPLE (first period, no turnover constraint)' if is_first else 'WITH TURNOVER CONSTRAINT'}")
        log(f"Target Long Weight: {float(target_weights[target_weights > 0].sum()):.4f}")
        log(f"Target Short Weight: {abs(float(target_weights[target_weights < 0].sum())):.4f}")
        log(f"Weight Turnover: {weight_turnover:.4f}")
        log(f"Target Weights (top 5 abs): {sorted(enumerate(target_weights), key=lambda x: abs(x[1]), reverse=True)[:5]}")

        # --- Execute trades at prices[t-1] ---
        target_stock_dollars: np.ndarray = target_weights * portfolio_value_start
        trade_dollars: np.ndarray = target_stock_dollars - stock_values
        shares = target_stock_dollars / prices[t - 1]
        cash_dollars -= float(trade_dollars.sum())

        stock_transaction_cost: float = float(np.abs(trade_dollars).sum()) * (tx_cost_bps / 10_000)
        cash_dollars -= stock_transaction_cost

        log(f"\n--- STOCK TRADES ---")
        log(f"Trade Dollars (total): ${float(np.abs(trade_dollars).sum()):,.2f}")
        log(f"Shares After: {shares[:5]}")
        log(f"Transaction Cost: ${stock_transaction_cost:,.2f}")
        log(f"Cash After Transaction Costs: ${cash_dollars:,.2f}")

        # --- Sell options after stock trades ---
        if options_enabled:
            options_result: dict = sell_options_overlay(
                shares, prices[t - 1], alphas, cov_matrix,
                call_otm_pct, put_otm_pct, call_alpha_barrier, put_alpha_barrier,
                cash_rate, contract_fee, spread_bps,
            )
            premium_collected: float = options_result["premium_collected"]
            active_options = options_result["option_positions"]
            cash_dollars += premium_collected

            log(f"\n--- OPTIONS SOLD ---")
            log(f"Contracts Sold: {options_result['num_contracts']}")
            log(f"Premium Collected: ${premium_collected:,.2f}")
            log(f"Cash After Options: ${cash_dollars:,.2f}")
            for opt in active_options[:3]:
                log(f"  Asset {opt['asset_idx']}: {opt['type']} @ strike {opt['strike']:.2f}")
        else:
            premium_collected = 0.0
            active_options = []

        log(f"\n--- HOLD DURING PERIOD ---")
        log(f"Market moves from prices[{t-1}] to prices[{t}]")

        # --- End of period: settle options, apply financing, record P&L ---
        stock_values_end: np.ndarray = shares * prices[t]

        log(f"\n--- END OF PERIOD (t={t}) ---")
        log(f"Stock Values After Market Move: ${float(stock_values_end.sum()):,.2f}")

        if options_enabled and active_options:
            options_settlement: float = settle_options(
                active_options, prices[t], contract_fee, spread_bps
            )
            cash_dollars += options_settlement
            options_net_income: float = premium_collected + options_settlement

            log(f"\n--- OPTIONS SETTLEMENT ---")
            log(f"Settlement Cost: ${options_settlement:,.2f}")
            log(f"Net Options Income: ${options_net_income:,.2f}")
            log(f"Cash After Settlement: ${cash_dollars:,.2f}")
        else:
            options_net_income = 0.0

        long_dollars: float = float(stock_values_end[stock_values_end > 0].sum())
        short_dollars: float = abs(float(stock_values_end[stock_values_end < 0].sum()))
        financing_cost: float = calculate_financing_cost(
            cash_dollars, long_dollars, cash_rate, margin_rate, borrow_fee, short_dollars
        )
        cash_dollars += financing_cost

        log(f"\n--- FINANCING ---")
        log(f"Long Positions: ${long_dollars:,.2f}")
        log(f"Short Positions: ${short_dollars:,.2f}")
        log(f"Financing Cost/Income: ${financing_cost:,.2f}")
        log(f"Cash After Financing: ${cash_dollars:,.2f}")

        portfolio_value_end: float = float(stock_values_end.sum()) + cash_dollars
        period_return: float = (portfolio_value_end - portfolio_value_start) / portfolio_value_start

        log(f"\n--- PERIOD SUMMARY ---")
        log(f"Portfolio Value End: ${portfolio_value_end:,.2f}")
        log(f"Period Return: {period_return * 100:.4f}%")
        log(f"Verification (Stock + Cash): ${float(stock_values_end.sum()) + cash_dollars:,.2f}")
        log(f"Discrepancy: ${portfolio_value_end - (float(stock_values_end.sum()) + cash_dollars):,.2e}")

        portfolio_returns.append(period_return)
        financing_costs.append(financing_cost / portfolio_value_start)
        portfolio_weights_history.append(target_weights.tolist())
        turnovers.append(weight_turnover)
        options_income.append(options_net_income / portfolio_value_start)

    # --- Summary statistics ---
    rets: np.ndarray = np.array(portfolio_returns)
    cum_ret: float = float(np.prod(1 + rets) - 1)
    n_months: int = len(rets)
    ann_ret: float = float((1 + cum_ret) ** (12 / n_months) - 1) if n_months > 0 else 0.0
    mean_m: float = float(rets.mean()) if n_months > 0 else 0.0
    std_m: float = float(rets.std()) if n_months > 0 else 0.0
    sharpe: float = float(mean_m / std_m * np.sqrt(12)) if std_m > 0 else 0.0

    avg_turnover: float = (
        float(np.mean(turnovers[1:])) if len(turnovers) > 1 else (turnovers[0] if turnovers else 0.0)
    )

    return {
        "portfolio_returns": [float(r * 100) for r in portfolio_returns],
        "financing_costs": [float(f * 100) for f in financing_costs],
        "portfolio_weights": portfolio_weights_history,
        "turnovers": [float(t * 100) for t in turnovers],
        "cumulative_return": cum_ret * 100,
        "annualized_return": ann_ret * 100,
        "sharpe_ratio": sharpe,
        "avg_turnover": avg_turnover * 100,
        "avg_financing_cost": float(np.mean(financing_costs)) * 100 * 12 if financing_costs else 0.0,
        "options_income": [float(o * 100) for o in options_income],
        "avg_options_income": float(np.mean(options_income)) * 100 * 12 if options_income else 0.0,
    }