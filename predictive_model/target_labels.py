# target_labels.py
import pandas as pd
import numpy as np

def create_target_label(
    df: pd.DataFrame,
    mode: str = "daily",
    min_return_threshold: float = None,
    forward_periods: int = 1
) -> pd.DataFrame:
    """
    Create cost-aware target labels for trading models.
    
    Args:
        df: DataFrame with OHLCV data
        mode: "daily" or "intraday"
        min_return_threshold: Minimum return to classify as BUY (default auto-set)
        forward_periods: How many bars ahead to predict (default 1)
    
    Returns:
        DataFrame with 'target', 'forward_return', and 'target_3class' columns
    """
    df = df.copy()
    
    # Auto-set thresholds based on mode
    if min_return_threshold is None:
        if mode == "daily":
            # Daily: 0.1% threshold (only need to beat commission)
            min_return_threshold = 0.0010
        else:
            # Intraday: tighter threshold
            min_return_threshold = 0.0005
    
    # Calculate forward return
    df["forward_return"] = df["Close"].pct_change(forward_periods).shift(-forward_periods)
    
    # Binary target: 1 if forward return exceeds threshold
    df["target"] = (df["forward_return"] > min_return_threshold).astype(int)
    
    # ✅ ALWAYS create 3-class target (for future multi-class models)
    df["target_3class"] = pd.cut(
        df["forward_return"],
        bins=[-np.inf, -min_return_threshold, min_return_threshold, np.inf],
        labels=[0, 1, 2]  # 0=SELL, 1=HOLD, 2=BUY
    )
    
    # ✅ Convert to float (handles NaN properly)
    df["target_3class"] = df["target_3class"].astype(float)
    
    return df


def create_regression_target(df: pd.DataFrame, mode: str = "daily") -> pd.DataFrame:
    """
    Alternative: Predict actual returns (regression) instead of classification.
    Allows for position sizing based on confidence.
    """
    df = df.copy()
    
    # Target = next bar's return
    df["target_return"] = df["Close"].pct_change().shift(-1)
    
    # Optional: Clip extreme outliers (prevent overfitting to black swans)
    if mode == "daily":
        lower, upper = df["target_return"].quantile([0.01, 0.99])
    else:
        lower, upper = df["target_return"].quantile([0.02, 0.98])
    
    df["target_return_clipped"] = df["target_return"].clip(lower, upper)
    
    return df


def backtest_threshold(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    returns: pd.Series,
    threshold_range: tuple = (0.50, 0.70),  # 👈 Lower upper bound
    min_trades: int = 15,  # 👈 NEW: Require at least 15 trades
    optimization_metric: str = "profit_factor"  # 👈 Switch from Sharpe
) -> dict:
    """
    Find optimal probability threshold based on trading profitability.
    
    Args:
        y_true: Actual labels
        y_pred_proba: Model predicted probabilities
        returns: Actual forward returns
        threshold_range: (min, max) thresholds to test
        min_trades: Minimum trades required for threshold to be valid
        optimization_metric: 'sharpe', 'profit_factor', or 'composite'
    """
    from sklearn.metrics import precision_score, recall_score
    
    best_score = -np.inf
    best_threshold = 0.5
    best_metrics = {}
    
    thresholds = np.arange(threshold_range[0], threshold_range[1] + 0.01, 0.05)
    
    for thresh in thresholds:
        y_pred = (y_pred_proba > thresh).astype(int)
        
        # Skip if no trades
        if y_pred.sum() == 0:
            continue
        
        # Financial metrics
        trades = y_pred == 1
        trade_returns = returns[trades]
        
        # ✅ ENFORCE MINIMUM TRADES
        if len(trade_returns) < min_trades:
            continue
        
        win_rate = (trade_returns > 0).mean()
        avg_return = trade_returns.mean()
        std_return = trade_returns.std()
        
        # Sharpe
        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Profit factor
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        
        if len(wins) > 0 and len(losses) > 0:
            gross_profit = wins.sum()
            gross_loss = abs(losses.sum())
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 0
        
        # ML metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # ✅ SELECT OPTIMIZATION METRIC
        if optimization_metric == "sharpe":
            score = sharpe
        elif optimization_metric == "profit_factor":
            score = profit_factor
        else:  # composite
            # Balance profit factor (60%) + win rate (20%) + recall (20%)
            # Ensures we take enough trades AND they're profitable
            score = (profit_factor * 0.6) + (win_rate * 0.2) + (recall * 0.2)
        
        # Track best
        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = {
                "threshold": thresh,
                "sharpe": sharpe,
                "profit_factor": profit_factor,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "n_trades": len(trade_returns),
                "precision": precision,
                "recall": recall,
                "score": score,
            }
    
    # ✅ FALLBACK: If no threshold meets min_trades, use 0.55
    if not best_metrics:
        print(f"[WARN] No threshold met min_trades={min_trades}. Falling back to 0.55")
        y_pred = (y_pred_proba > 0.55).astype(int)
        trades = y_pred == 1
        trade_returns = returns[trades]
        
        if len(trade_returns) > 0:
            win_rate = (trade_returns > 0).mean()
            avg_return = trade_returns.mean()
            std_return = trade_returns.std()
            sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            
            wins = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns < 0]
            profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else 0
            
            best_metrics = {
                "threshold": 0.55,
                "sharpe": sharpe,
                "profit_factor": profit_factor,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "n_trades": len(trade_returns),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "score": 0,
            }
    
    return best_metrics
