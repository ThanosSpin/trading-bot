# trading_metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_financial_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    returns: np.ndarray,
    initial_capital: float = 1000.0,
    commission: float = 0.001,  # 0.1% per trade
) -> dict:
    """
    Calculate trading-specific metrics that matter for profitability.
    
    Args:
        y_true: Actual labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_pred_proba: Predicted probabilities (0 to 1)
        returns: Actual forward returns per bar
        initial_capital: Starting capital
        commission: Round-trip commission as decimal (0.001 = 0.1%)
    
    Returns:
        Dict with financial and ML metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Standard ML metrics
    ml_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Filter to only trades where model predicted BUY
    trades_mask = y_pred == 1
    trade_returns = returns[trades_mask]
    
    # ✅ HANDLE CASE: No trades
    if len(trade_returns) == 0:
        return {
            **ml_metrics,
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "final_capital": initial_capital,
            "largest_win": 0.0,
            "largest_loss": 0.0,
        }
    
    # Apply commission
    trade_returns_after_cost = trade_returns - commission
    
    # Win/Loss analysis
    wins = trade_returns_after_cost[trade_returns_after_cost > 0]
    losses = trade_returns_after_cost[trade_returns_after_cost <= 0]
    
    win_rate = len(wins) / len(trade_returns_after_cost) if len(trade_returns_after_cost) > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0  # Will be negative
    
    # Profit factor
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    
    # Risk-adjusted metrics
    sharpe_ratio = 0.0
    if len(trade_returns_after_cost) > 1 and trade_returns_after_cost.std() > 0:
        sharpe_ratio = float((trade_returns_after_cost.mean() / trade_returns_after_cost.std()) * np.sqrt(252))
    
    # Drawdown calculation
    cumulative_returns = (1 + trade_returns_after_cost).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    # Total return
    if isinstance(cumulative_returns, pd.Series):
        total_return = float(cumulative_returns.iloc[-1] - 1)
    else:
        total_return = float(cumulative_returns[-1] - 1)
    
    final_capital = initial_capital * (1 + total_return)
    
    # Largest win/loss
    largest_win = float(wins.max()) if len(wins) > 0 else 0.0
    largest_loss = float(losses.min()) if len(losses) > 0 else 0.0
    
    return {
        **ml_metrics,
        "n_trades": int(len(trade_returns)),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "total_return": float(total_return),
        "final_capital": float(final_capital),
        "largest_win": float(largest_win),
        "largest_loss": float(largest_loss),
    }

def print_trading_report(metrics: dict):
    """Pretty print trading metrics."""
    
    print("\n" + "="*50)
    print("📊 ML METRICS")
    print("="*50)
    print(f"Accuracy:     {metrics.get('accuracy', 0):.2%}")
    print(f"Precision:    {metrics.get('precision', 0):.2%}")
    print(f"Recall:       {metrics.get('recall', 0):.2%}")
    print(f"F1 Score:     {metrics.get('f1', 0):.4f}")
    
    print("\n" + "="*50)
    print("💰 FINANCIAL METRICS")
    print("="*50)
    
    n_trades = metrics.get('n_trades', 0)
    
    # ✅ Handle zero trades case
    if n_trades == 0:
        print("⚠️  NO TRADES GENERATED - Model threshold too high or no buy signals")
        print("="*50 + "\n")
        return
    
    print(f"Total Trades:      {n_trades}")
    print(f"Win Rate:          {metrics.get('win_rate', 0):.2%}")
    
    pf = metrics.get('profit_factor', 0)
    if pf == np.inf:
        print(f"Profit Factor:     ∞ (no losses)")
    else:
        print(f"Profit Factor:     {pf:.2f}")
    
    print(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown:      {metrics.get('max_drawdown', 0):.2%}")
    print(f"Total Return:      {metrics.get('total_return', 0):.2%}")
    print(f"Final Capital:     ${metrics.get('final_capital', 1000):,.2f}")
    
    print("\n" + "="*50)
    print("📈 WIN/LOSS BREAKDOWN")
    print("="*50)
    print(f"Avg Win:           {metrics.get('avg_win', 0):.2%}")
    print(f"Avg Loss:          {metrics.get('avg_loss', 0):.2%}")
    print(f"Largest Win:       {metrics.get('largest_win', 0):.2%}")
    print(f"Largest Loss:      {metrics.get('largest_loss', 0):.2%}")
    print("="*50 + "\n")
