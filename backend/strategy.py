import numpy as np
import logging
from typing import Optional
from datetime import datetime
from backend.alpha import calculate_alphas
from backend.risk import estimate_covariance_ewma
from backend.optimizer import optimize_portfolio, optimize_portfolio_simple
from backend.options import sell_options_overlay, settle_options

MONTHS_PER_YEAR = 12
INITIAL_CAPITAL = 1_000_000.0


def setup_logger(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logger for strategy execution"""
    if not verbose:
        return None
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"strategy_log_{timestamp}.txt"
    
    logger = logging.getLogger('strategy')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def calculate_financing_cost(cash_dollars: float, long_dollars: float,
                             cash_rate: float, margin_rate: float,
                             borrow_fee: float, short_dollars: float) -> float:
    """Calculate monthly financing cost/income based on dollar positions"""
    monthly_cash_rate = cash_rate / MONTHS_PER_YEAR
    monthly_margin_rate = margin_rate / MONTHS_PER_YEAR
    monthly_borrow_fee = borrow_fee / MONTHS_PER_YEAR
    
    if cash_dollars >= 0:
        cash_income = cash_dollars * monthly_cash_rate
        margin_cost = 0
    else:
        cash_income = 0
        margin_cost = abs(cash_dollars) * monthly_margin_rate
    
    borrow_cost = short_dollars * monthly_borrow_fee
    net_financing = cash_income - margin_cost - borrow_cost
    
    return net_financing


def clean_small_weights(weights: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """
    Zero out tiny weights and rescale longs/shorts to preserve total exposure.
    """
    cleaned = weights.copy()
    
    # Store original total exposures
    original_long_total = np.sum(cleaned[cleaned > 0])
    original_short_total = np.sum(cleaned[cleaned < 0])
    
    # Zero out small weights
    cleaned[np.abs(cleaned) < threshold] = 0.0
    
    # Rescale longs to maintain total long exposure
    if np.sum(cleaned > 0) > 0:
        new_long_total = np.sum(cleaned[cleaned > 0])
        if new_long_total > 0:
            cleaned[cleaned > 0] *= (original_long_total / new_long_total)
    
    # Rescale shorts to maintain total short exposure
    if np.sum(cleaned < 0) > 0:
        new_short_total = np.sum(cleaned[cleaned < 0])
        if new_short_total < 0:
            cleaned[cleaned < 0] *= (original_short_total / new_short_total)
    
    return cleaned


def run_strategy(returns: np.ndarray,
                prices: np.ndarray,
                factor_scores: np.ndarray,
                strategy_params: dict[str, float],
                financing_params: dict[str, float],
                options_params: dict,
                verbose: bool = False,
                log_file: Optional[str] = None) -> dict:
    """
    Run the long-short strategy over a window of returns and prices.

    At each step:
    - optimize and trade at prices[t-1]
    - hold positions as prices move to prices[t]
    - settle options, apply financing, and record P&L
    """
    logger = setup_logger(verbose, log_file)
    
    if logger:
        logger.info("=" * 80)
        logger.info("STRATEGY EXECUTION LOG")
        logger.info("=" * 80)
        logger.info(f"\nInitial Capital: ${INITIAL_CAPITAL:,.2f}")
        logger.info(f"Strategy Parameters: {strategy_params}")
        logger.info(f"Financing Parameters: {financing_params}")
        logger.info(f"Options Parameters: {options_params}")
        logger.info("=" * 80 + "\n")

    # Extract parameters
    lookback = int(strategy_params['lookback'])
    strategy_length = int(strategy_params['strategy_length'])
    risk_aversion = strategy_params['risk_aversion']
    target_long_weight = strategy_params['long_weight'] / 100
    target_short_weight = strategy_params['short_weight'] / 100
    turnover_limit = strategy_params['turnover_limit'] / 100
    max_weight = strategy_params['max_weight'] / 100
    transaction_cost_bps = strategy_params['transaction_cost_bps']
    
    cash_rate = financing_params['cash_rate'] / 100
    margin_rate = financing_params['margin_rate'] / 100
    borrow_fee = financing_params['borrow_fee'] / 100

    # Extract options parameters
    options_enabled = options_params.get('enabled', False)
    call_otm_pct = options_params.get('call_otm_pct', 5.0)
    put_otm_pct = options_params.get('put_otm_pct', 5.0)
    call_alpha_barrier = options_params.get('call_alpha_barrier', 0.0)
    put_alpha_barrier = options_params.get('put_alpha_barrier', 0.0)
    contract_fee = options_params.get('contract_fee', 0.65)
    spread_bps = options_params.get('spread_bps', 10.0)
    
    net_exposure = target_long_weight - target_short_weight
    cash_exposure = 1.0 - net_exposure
    
    n_periods, n_assets = returns.shape
    
    # Tracking
    portfolio_returns: list[float] = []
    financing_costs: list[float] = []
    portfolio_weights_history: list[list[float]] = []
    turnovers: list[float] = []
    options_income: list[float] = []
    
    # Initialize at t=0
    cash_dollars = INITIAL_CAPITAL
    shares = np.zeros(n_assets)
    active_options = []
    
    # Loop through periods: [lookback→lookback+1], [lookback+1→lookback+2], ...
    for t in range(lookback + 1, lookback + strategy_length + 1):
        if t >= n_periods + 1:
            break

        period_num = t - lookback
        if logger:
            logger.info("\n" + "=" * 80)
            logger.info(f"PERIOD {period_num}: [{t-1} → {t}]")
            logger.info("=" * 80)
        
        # BEGINNING OF PERIOD [t-1 → t]: optimize and trade at t-1
        stock_values = shares * prices[t-1]
        portfolio_value_start = np.sum(stock_values) + cash_dollars
        current_weights = stock_values / portfolio_value_start if portfolio_value_start > 0 else np.zeros(n_assets)

        if logger:
            logger.info(f"\n--- START OF PERIOD (t={t-1}) ---")
            logger.info(f"Portfolio Value: ${portfolio_value_start:,.2f}")
            logger.info(f"Cash: ${cash_dollars:,.2f}")
            logger.info(f"Stock Positions: ${np.sum(stock_values):,.2f}")
            logger.info(f"Long Positions: ${np.sum(stock_values[stock_values > 0]):,.2f}")
            logger.info(f"Short Positions: ${np.sum(stock_values[stock_values < 0]):,.2f}")
            logger.info(f"Current Weights (top 5 abs): {sorted(enumerate(current_weights), key=lambda x: abs(x[1]), reverse=True)[:5]}")
        
        # Estimate covariance from returns ending at t-1
        returns_window = returns[t-1-lookback:t-1]
        cov_matrix = estimate_covariance_ewma(returns_window)
        
        # Calculate alphas using factor scores at t-2
        alphas = calculate_alphas(factor_scores[t-1], cov_matrix)

        if logger:
            logger.info(f"\nAlphas (top 5): {sorted(enumerate(alphas), key=lambda x: x[1], reverse=True)[:5]}")
            logger.info(f"Alphas (bottom 5): {sorted(enumerate(alphas), key=lambda x: x[1])[:5]}")
        
        # Optimize (first period: no turnover constraint)
        if t == lookback + 1:
            target_weights = optimize_portfolio_simple(
                alphas, cov_matrix, risk_aversion,
                target_long_weight, target_short_weight, max_weight
            )
            target_weights = clean_small_weights(target_weights)
            weight_turnover = np.sum(np.abs(target_weights))
            if logger:
                logger.info(f"\nOptimization: SIMPLE (first period, no turnover constraint)")
        else:
            result = optimize_portfolio(
                alphas, cov_matrix, current_weights, risk_aversion,
                target_long_weight, target_short_weight, turnover_limit, max_weight
            )
            target_weights = result['weights']
            target_weights = clean_small_weights(target_weights)
            weight_turnover = result['turnover']
            if logger:
                logger.info(f"\nOptimization: WITH TURNOVER CONSTRAINT")

        if logger:
            logger.info(f"Target Long Weight: {np.sum(target_weights[target_weights > 0]):.4f}")
            logger.info(f"Target Short Weight: {abs(np.sum(target_weights[target_weights < 0])):.4f}")
            logger.info(f"Weight Turnover: {weight_turnover:.4f}")
            logger.info(f"Target Weights (top 5 abs): {sorted(enumerate(target_weights), key=lambda x: abs(x[1]), reverse=True)[:5]}")
        
        # Execute trades at prices[t-1]
        target_stock_dollars = target_weights * portfolio_value_start
        trade_dollars = target_stock_dollars - stock_values
        shares_old = shares.copy()
        shares = target_stock_dollars / prices[t-1]
        cash_dollars -= np.sum(trade_dollars)

        if logger:
            logger.info(f"\n--- STOCK TRADES ---")
            logger.info(f"Trade Dollars (total): ${np.sum(np.abs(trade_dollars)):,.2f}")
            logger.info(f"Shares Before: {shares_old[:5]}")
            logger.info(f"Shares After: {shares[:5]}")
            logger.info(f"Cash After Trades: ${cash_dollars:,.2f}")

        # Apply stock transaction costs
        stock_transaction_cost = np.sum(np.abs(trade_dollars)) * (transaction_cost_bps / 10000)
        cash_dollars -= stock_transaction_cost
        
        if logger:
            logger.info(f"Transaction Cost: ${stock_transaction_cost:,.2f}")
            logger.info(f"Cash After Transaction Costs: ${cash_dollars:,.2f}")

        # SELL OPTIONS after stock trades
        if options_enabled:
            options_result = sell_options_overlay(
                shares, prices[t-1], alphas, cov_matrix,
                call_otm_pct, put_otm_pct, call_alpha_barrier, put_alpha_barrier,
                cash_rate, contract_fee, spread_bps
            )
            premium_collected = options_result['premium_collected']
            active_options = options_result['option_positions']
            cash_dollars += premium_collected

            if logger:
                logger.info(f"\n--- OPTIONS SOLD ---")
                logger.info(f"Contracts Sold: {options_result['num_contracts']}")
                logger.info(f"Premium Collected: ${premium_collected:,.2f}")
                logger.info(f"Cash After Options: ${cash_dollars:,.2f}")
                for opt in active_options[:3]:  # Show first 3
                    logger.info(f"  Asset {opt['asset_idx']}: {opt['type']} @ strike {opt['strike']:.2f}")
        else:
            premium_collected = 0.0
            active_options = []

        if logger:
            logger.info(f"\n--- HOLD DURING PERIOD ---")
            logger.info(f"Market moves from prices[{t-1}] to prices[{t}]")
        
        # HOLD POSITIONS DURING PERIOD: market moves from t-1 to t
        
        # END OF PERIOD at t: settle options, financing, calculate returns
        stock_values_end = shares * prices[t]

        if logger:
            logger.info(f"\n--- END OF PERIOD (t={t}) ---")
            logger.info(f"Stock Values After Market Move: ${np.sum(stock_values_end):,.2f}")

        # Settle expired options
        if options_enabled and active_options:
            options_settlement = settle_options(active_options, prices[t], contract_fee, spread_bps)
            cash_dollars += options_settlement
            options_net_income = premium_collected + options_settlement
            
            if logger:
                logger.info(f"\n--- OPTIONS SETTLEMENT ---")
                logger.info(f"Settlement Cost: ${options_settlement:,.2f}")
                logger.info(f"Net Options Income: ${options_net_income:,.2f}")
                logger.info(f"Cash After Settlement: ${cash_dollars:,.2f}")
        else:
            options_net_income = 0.0
        
        # Accrue financing
        long_dollars = np.sum(stock_values_end[stock_values_end > 0])
        short_dollars = abs(np.sum(stock_values_end[stock_values_end < 0]))
        financing_cost = calculate_financing_cost(
            cash_dollars, long_dollars, cash_rate,
            margin_rate, borrow_fee, short_dollars
        )
        cash_dollars += financing_cost
        
        if logger:
            logger.info(f"\n--- FINANCING ---")
            logger.info(f"Long Positions: ${long_dollars:,.2f}")
            logger.info(f"Short Positions: ${short_dollars:,.2f}")
            logger.info(f"Financing Cost/Income: ${financing_cost:,.2f}")
            logger.info(f"Cash After Financing: ${cash_dollars:,.2f}")
        
        # Calculate period return
        portfolio_value_end = np.sum(stock_values_end) + cash_dollars
        period_return = (portfolio_value_end - portfolio_value_start) / portfolio_value_start

        if logger:
            logger.info(f"\n--- PERIOD SUMMARY ---")
            logger.info(f"Portfolio Value End: ${portfolio_value_end:,.2f}")
            logger.info(f"Period Return: {period_return*100:.4f}%")
            logger.info(f"Verification (Stock + Cash): ${np.sum(stock_values_end) + cash_dollars:,.2f}")
            logger.info(f"Discrepancy: ${portfolio_value_end - (np.sum(stock_values_end) + cash_dollars):,.2e}")
        
        # Record
        portfolio_returns.append(period_return)
        financing_costs.append(financing_cost / portfolio_value_start)
        portfolio_weights_history.append(target_weights.tolist())
        turnovers.append(weight_turnover)
        options_income.append(options_net_income / portfolio_value_start)
    
    # Calculate statistics
    returns_array = np.array(portfolio_returns)
    cumulative_return = np.prod(1 + returns_array) - 1
    n_months = len(returns_array)

    annualized_return = (
        (1 + cumulative_return) ** (12 / n_months) - 1
        if n_months > 0 else 0.0
    )
        
    mean_monthly = np.mean(returns_array) if n_months > 0 else 0.0
    std_monthly = np.std(returns_array) if n_months > 0 else 0.0
    sharpe = (mean_monthly / std_monthly * np.sqrt(12)) if std_monthly > 0 else 0.0
    
    avg_turnover = np.mean(turnovers[1:]) if len(turnovers) > 1 else (turnovers[0] if turnovers else 0.0)
    avg_financing = np.mean(financing_costs) if financing_costs else 0.0
    avg_options_income = np.mean(options_income) if options_income else 0.0
    
    return {
        'portfolio_returns': [float(r * 100) for r in portfolio_returns],
        'financing_costs': [float(f * 100) for f in financing_costs],
        'portfolio_weights': portfolio_weights_history,
        'turnovers': [float(t * 100) for t in turnovers],
        'cumulative_return': float(cumulative_return * 100),
        'annualized_return': float(annualized_return * 100),
        'sharpe_ratio': float(sharpe),
        'avg_turnover': float(avg_turnover * 100),
        'avg_financing_cost': float(avg_financing * 100 * 12),
        'options_income': [float(o * 100) for o in options_income],
        'avg_options_income': float(avg_options_income * 100 * 12),
    }
