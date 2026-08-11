# strategy.py
"""Factor-based portfolio strategy with self-financing cash accounting.

NAV identity (always):
    NAV = Σ stock_dollars + cash
After a rebalance to weights w (pre-cost):
    stock_dollars = w · NAV
    cash          = (1 − 1'w) · NAV
so cash is the budget residual — nothing is created or destroyed except
explicit P&L (tx costs, option premium/settlement, financing).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np

from backend.alpha import calculate_alphas
from backend.risk import estimate_covariance
from backend.optimizer import PortfolioOptimizer, optimize_portfolio
from backend.options import sell_options_overlay, settle_options

MONTHS_PER_YEAR = 12
INITIAL_CAPITAL = 1_000_000.0
_EPS = 1e-9


def setup_logger(verbose: bool = False, log_file: Optional[str] = None) -> Optional[logging.Logger]:
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


def calculate_financing(
    cash_dollars: float,
    short_dollars: float,
    cash_rate: float,
    margin_rate: float,
    borrow_fee: float,
) -> float:
    """
    Monthly financing P&L on capital committed over the hold.

    - Positive cash earns cash_rate (includes residual cash & short proceeds).
    - Negative cash (margin debit from leverage) pays margin_rate.
    - Short notional pays borrow_fee.
    """
    if cash_dollars >= 0:
        cash_pnl = cash_dollars * (cash_rate / MONTHS_PER_YEAR)
    else:
        cash_pnl = cash_dollars * (margin_rate / MONTHS_PER_YEAR)  # more negative
    borrow_cost = short_dollars * (borrow_fee / MONTHS_PER_YEAR)
    return cash_pnl - borrow_cost


def clean_small_weights(
    weights: np.ndarray,
    long_target: float,
    short_target: float,
    threshold: float = 1e-4,
) -> np.ndarray:
    """Zero tiny positions, then rescale sleeves back to target gross exposures."""
    w = np.asarray(weights, dtype=float).copy()
    w[np.abs(w) < threshold] = 0.0

    long_sum = float(w[w > 0].sum())
    short_sum = float(-w[w < 0].sum())

    if long_target > 0 and long_sum > _EPS:
        w[w > 0] *= long_target / long_sum
    elif long_target <= _EPS:
        w[w > 0] = 0.0

    if short_target > 0 and short_sum > _EPS:
        w[w < 0] *= short_target / short_sum
    elif short_target <= _EPS:
        w[w < 0] = 0.0

    return w


def run_strategy(
    returns: np.ndarray,
    prices: np.ndarray,
    factor_scores: np.ndarray,
    strategy_params: dict,
    financing_params: dict,
    options_params: dict,
    verbose: bool = False,
    log_file: Optional[str] = None,
    optimizer: Optional[PortfolioOptimizer] = None,
) -> dict:
    """
    Run the factor strategy over a window of returns and prices.

    Each month:
      1. Mark NAV at t-1
      2. Estimate Σ and alphas; optimize weights
      3. Rebalance (self-financing); pay tx costs from cash
      4. Optionally sell 1M options; premium → cash
      5. Hold to t (price move)
      6. Settle options; accrue financing on post-trade holdings; mark NAV
    """
    logger = setup_logger(verbose, log_file)
    log = logger.info if logger else lambda *a, **k: None

    lookback = int(strategy_params["lookback"])
    strategy_length = int(strategy_params["strategy_length"])
    risk_aversion = float(strategy_params["risk_aversion"])
    target_long = float(strategy_params["long_weight"])
    target_short = float(strategy_params["short_weight"])
    max_long = float(strategy_params.get("max_long_weight", strategy_params.get("max_weight", 0.1)))
    max_short = float(strategy_params.get("max_short_weight", max_long))
    tx_cost_bps = float(strategy_params["transaction_cost_bps"])
    market_impact_coef = float(strategy_params.get("market_impact_coef", 0.0))
    hard_turnover_limit = float(
        strategy_params.get(
            "hard_turnover_limit",
            strategy_params.get("turnover_limit", 0.0),
        )
    )
    weight_threshold = float(strategy_params.get("weight_threshold", 1e-4))
    signal_ic = float(strategy_params.get("signal_ic", 0.05))
    alpha_method = str(strategy_params.get("alpha_method", "grinold"))
    cov_method = str(strategy_params.get("cov_method", "ewma"))
    cov_halflife = int(strategy_params.get("cov_halflife", lookback))
    # κ₁: same units as cash TC — fraction of NAV per unit Σ|Δw|
    tc_linear = tx_cost_bps / 10_000.0
    tc_quad = market_impact_coef

    cash_rate = float(financing_params["cash_rate"])
    margin_rate = float(financing_params["margin_rate"])
    borrow_fee = float(financing_params["borrow_fee"])

    options_enabled = bool(options_params.get("enabled", False))
    call_otm_pct = float(options_params.get("call_otm_pct", 0.01))
    put_otm_pct = float(options_params.get("put_otm_pct", 0.01))
    call_alpha_barrier = float(options_params["call_alpha_barrier"])
    put_alpha_barrier = float(options_params["put_alpha_barrier"])
    contract_fee = float(options_params.get("contract_fee", 0.65))
    spread_bps = float(options_params.get("spread_bps", 25.0))
    dividend_yield = float(options_params.get("dividend_yield", 0.0))

    n_periods, n_assets = returns.shape
    if optimizer is None:
        optimizer = PortfolioOptimizer(
            n_assets,
            risk_aversion,
            target_long,
            target_short,
            max_long,
            max_short,
            hard_turnover_limit,
            tc_linear=tc_linear,
            tc_quad=tc_quad,
        )

    portfolio_returns: list[float] = []
    financing_costs: list[float] = []
    portfolio_weights_history: list[list[float]] = []
    turnovers: list[float] = []
    options_income: list[float] = []

    cash = float(INITIAL_CAPITAL)
    shares = np.zeros(n_assets)
    active_options: list[dict] = []

    log("=" * 80)
    log("STRATEGY EXECUTION LOG")
    log(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    log(f"Targets: long={target_long:.4f} short={target_short:.4f} "
        f"net={target_long - target_short:.4f} "
        f"cash_residual={1.0 - (target_long - target_short):.4f}")
    log(f"Strategy: {strategy_params}")
    log("=" * 80)

    for t in range(lookback + 1, lookback + strategy_length + 1):
        if t >= n_periods + 1:
            break

        period_num = t - lookback
        px = prices[t - 1]
        stock_values = shares * px
        nav = float(stock_values.sum() + cash)

        if nav <= _EPS:
            log(f"Period {period_num}: NAV={nav:.2f} ruined — stop.")
            break

        # Numerical hygiene: force residual cash identity at mark
        cash = nav - float(stock_values.sum())
        current_weights = stock_values / nav

        log(f"\nPERIOD {period_num}: t={t - 1}->{t}  NAV=${nav:,.2f}  cash=${cash:,.2f}")

        cov = estimate_covariance(
            returns[t - 1 - lookback : t - 1],
            method=cov_method,
            halflife=cov_halflife,
        )
        alphas = calculate_alphas(
            factor_scores[t - 1], cov, method=alpha_method, signal_ic=signal_ic
        )

        is_first = t == lookback + 1
        # Hard turnover caps apply after the book exists; founding is cost-penalized only.
        result = optimize_portfolio(
            alphas,
            cov,
            risk_aversion,
            target_long,
            target_short,
            max_long,
            max_short,
            current_weights=None if is_first else current_weights,
            tc_linear=tc_linear,
            tc_quad=tc_quad,
            hard_turnover_limit=0.0 if is_first else hard_turnover_limit,
            optimizer=None if is_first else optimizer,
        )
        target_w = clean_small_weights(
            result["weights"], target_long, target_short, weight_threshold
        )
        turnover = float(result["turnover"])
        if is_first:
            turnover = float(np.abs(target_w).sum())

        # --- Self-financing rebalance ---
        # stock' = w · NAV, cash' = NAV − 1'stock' − costs  (residual budget)
        target_stock = target_w * nav
        trade = target_stock - stock_values
        shares = target_stock / px
        tx_cost = float(np.abs(trade).sum()) * (tx_cost_bps / 10_000.0)
        cash = nav - float(target_stock.sum()) - tx_cost

        # Post-trade holdings used for financing over the month
        stock_hold = shares * px
        short_hold = float(-stock_hold[stock_hold < 0].sum())
        cash_hold = cash  # before option premium; premium is also held as cash
        nav_after_trade = float(stock_hold.sum() + cash)
        assert abs(nav_after_trade + tx_cost - nav) <= max(1e-6 * abs(nav), 1e-4), (
            f"budget break: nav={nav} after={nav_after_trade} cost={tx_cost}"
        )

        premium = 0.0
        if options_enabled:
            opt = sell_options_overlay(
                shares, px, alphas, cov,
                call_otm_pct, put_otm_pct, call_alpha_barrier, put_alpha_barrier,
                cash_rate, contract_fee, spread_bps,
                dividend_yield=dividend_yield,
            )
            premium = float(opt["premium_collected"])
            active_options = opt["option_positions"]
            cash += premium
            cash_hold = cash
        else:
            active_options = []

        # --- Hold: prices move to t ---
        stock_end = shares * prices[t]

        opt_settle = 0.0
        if options_enabled and active_options:
            opt_settle = float(
                settle_options(active_options, prices[t], contract_fee=contract_fee)
            )
            cash += opt_settle
        options_net = premium + opt_settle

        # Financing on capital committed during the hold (post-trade)
        fin = calculate_financing(cash_hold, short_hold, cash_rate, margin_rate, borrow_fee)
        cash += fin

        nav_end = float(stock_end.sum() + cash)
        period_ret = (nav_end - nav) / nav
        if nav_end <= _EPS:
            period_ret = -1.0
            nav_end = 0.0
            cash = 0.0
            shares[:] = 0.0

        log(
            f"  turnover={turnover:.4f} tx=${tx_cost:,.2f} opt_net=${options_net:,.2f} "
            f"fin=${fin:,.2f} NAV_end=${nav_end:,.2f} r={period_ret * 100:.4f}%"
        )

        portfolio_returns.append(period_ret)
        financing_costs.append(fin / nav)
        portfolio_weights_history.append(target_w.tolist())
        turnovers.append(turnover)
        options_income.append(options_net / nav)

        if period_ret <= -1.0:
            break

    rets = np.asarray(portfolio_returns, dtype=float) if portfolio_returns else np.array([])
    cum_ret = float(np.prod(1.0 + rets) - 1.0) if len(rets) else 0.0
    n_months = len(rets)
    if n_months > 0 and (1.0 + cum_ret) > 0:
        ann_ret = float((1.0 + cum_ret) ** (12 / n_months) - 1.0)
    else:
        ann_ret = -1.0 if n_months > 0 else 0.0
    mean_m = float(rets.mean()) if n_months else 0.0
    std_m = float(rets.std()) if n_months else 0.0
    sharpe = float(mean_m / std_m * np.sqrt(12)) if std_m > 0 else 0.0
    avg_turnover = (
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
