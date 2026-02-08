# feature_selection.py
import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ Auto-detect models directory
try:
    from model_xgb import MODEL_DIR
except ImportError:
    try:
        from config import MODEL_DIR
    except ImportError:
        MODEL_DIR = "models"


def get_optimal_feature_count(n_samples: int, n_features: int, mode: str = "daily") -> int:
    """
    Calculate optimal feature count based on data characteristics.
    
    Rule of thumb: 
    - Keep 30-50% of features
    - Ensure at least 10:1 sample-to-feature ratio
    - Mode-specific bounds
    
    Args:
        n_samples: Number of training samples
        n_features: Total available features
        mode: 'daily', 'intraday', 'intraday_mr', or 'intraday_mom'
    
    Returns:
        Optimal number of features to keep
    """
    # Sample-to-feature ratio (should be at least 10:1 for generalization)
    max_features_by_samples = n_samples // 10
    
    # Percentage of total features (keep 30-50%)
    min_features_pct = int(n_features * 0.30)
    max_features_pct = int(n_features * 0.50)
    
    # Mode-specific bounds
    if mode in ("intraday", "intraday_mr", "intraday_mom"):
        # Intraday needs more features (time-of-day, regime transitions)
        lower_bound = 30
        upper_bound = 60
    else:  # daily
        # Daily can use fewer (no intraday patterns)
        lower_bound = 25
        upper_bound = 50
    
    # Calculate optimal (conservative approach)
    optimal = min(max_features_pct, max_features_by_samples)
    optimal = max(optimal, min_features_pct)
    
    # Apply bounds
    optimal = max(optimal, lower_bound)
    optimal = min(optimal, upper_bound)
    
    # Never exceed total features
    optimal = min(optimal, n_features)
    
    return optimal


def select_features_with_shap(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    top_n: int = None,  # ✅ CHANGED: None = auto-calculate
    plot: bool = True,
    symbol: str = "UNKNOWN",
    mode: str = "daily"
) -> tuple:
    """
    Use SHAP to select most important features.
    
    ✨ ENHANCED: Auto-calculates optimal feature count if top_n=None
    
    Args:
        model: Trained XGBoost model
        X_train: Training features
        X_test: Test features
        top_n: Number of top features to keep (None = auto-calculate based on data)
        plot: Whether to generate SHAP plots
        symbol: Stock symbol (e.g., "NVDA") for filename
        mode: Model type (e.g., "daily", "intraday_mr", "intraday_mom")
    
    Returns:
        (top_features_list, shap_values_test, X_test_reduced)
    """
    n_samples, n_features = X_train.shape
    
    # ✅ AUTO-CALCULATE optimal feature count if not specified
    if top_n is None:
        top_n = get_optimal_feature_count(n_samples, n_features, mode)
        print(f"\n🔍 SHAP Feature Selection (AUTO-CALCULATED)")
        print(f"   Total features: {n_features}")
        print(f"   Training samples: {n_samples}")
        print(f"   Mode: {mode}")
        print(f"   Selected: {top_n} features ({top_n/n_features*100:.1f}%)")
    else:
        print(f"\n🔍 SHAP Feature Selection (MANUAL)")
        print(f"   Total features: {n_features}")
        print(f"   Selecting top {top_n} features")
    
    # ✅ Validate top_n
    if top_n > n_features:
        print(f"⚠️ Warning: top_n={top_n} > n_features={n_features}, using all features")
        top_n = n_features
    elif top_n < 10:
        print(f"⚠️ Warning: top_n={top_n} is very low, minimum 10 recommended")
        top_n = max(top_n, 10)
    
    # ✅ Ensure models/ directory exists
    model_dir = MODEL_DIR if MODEL_DIR else "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create SHAP explainer
    print(f"\n[SHAP] Creating TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values on test set (smaller, faster)
    print(f"[SHAP] Computing SHAP values on {len(X_test)} test samples...")
    shap_values = explainer.shap_values(X_test)
    
    # Get feature importance
    print(f"[SHAP] Calculating feature importance...")
    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "shap_importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("shap_importance", ascending=False)
    
    # Select top N features
    top_features = feature_importance.head(top_n)["feature"].tolist()
    
    # ✅ ENHANCED: Show feature importance statistics
    total_importance = feature_importance["shap_importance"].sum()
    top_n_importance = feature_importance.head(top_n)["shap_importance"].sum()
    coverage = top_n_importance / total_importance * 100
    
    print(f"\n✅ Selected Top {top_n} Features (covering {coverage:.1f}% of total importance)")
    print("=" * 70)
    print(f"{'Rank':<6} {'Feature':<35} {'SHAP Importance':<15} {'Cumulative %'}")
    print("-" * 70)
    
    cumulative_importance = 0
    for i, row in feature_importance.head(top_n).iterrows():
        cumulative_importance += row['shap_importance']
        cumulative_pct = cumulative_importance / total_importance * 100
        rank = feature_importance.index.get_loc(i) + 1
        print(f"{rank:<6} {row['feature']:<35} {row['shap_importance']:<15.6f} {cumulative_pct:>6.1f}%")
    
    print("=" * 70)
    
    # ✅ WARNING: Check if top features are too concentrated
    top_10_importance = feature_importance.head(10)["shap_importance"].sum()
    top_10_pct = top_10_importance / total_importance * 100
    
    if top_10_pct > 80:
        print(f"\n⚠️ WARNING: Top 10 features account for {top_10_pct:.1f}% of importance")
        print(f"   Consider reducing feature count or checking for redundant features")
    
    # Optionally plot
    if plot:
        try:
            # ✅ Descriptive filenames in models/ directory
            summary_filename = f"{symbol}_{mode}_shap_summary_top{top_n}.png"
            importance_filename = f"{symbol}_{mode}_shap_importance_top{top_n}.png"
            
            summary_path = os.path.join(model_dir, summary_filename)
            importance_path = os.path.join(model_dir, importance_filename)
            
            print(f"\n[SHAP] Generating plots...")
            
            # Summary plot (beeswarm)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_test,
                max_display=min(top_n, 30),  # Limit display to 30 for readability
                show=False
            )
            plt.title(f"SHAP Feature Summary - {symbol} {mode.upper()} (Top {top_n})", 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(summary_path, dpi=150, bbox_inches="tight")
            print(f"   ✅ Summary plot: {os.path.abspath(summary_path)}")
            plt.close()
            
            # Bar plot of importance
            fig, ax = plt.subplots(figsize=(12, 10))
            display_count = min(30, top_n)  # Show top 30 max
            top_display = feature_importance.head(display_count)
            
            # Color bars by importance (gradient)
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_display)))
            
            bars = ax.barh(range(len(top_display)), top_display["shap_importance"], color=colors)
            ax.set_yticks(range(len(top_display)))
            ax.set_yticklabels(top_display["feature"], fontsize=9)
            ax.set_xlabel("Mean |SHAP value|", fontsize=11)
            ax.set_title(f"Feature Importance - {symbol} {mode.upper()} (Top {display_count})", 
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, top_display["shap_importance"])):
                ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(importance_path, dpi=150, bbox_inches="tight")
            print(f"   ✅ Importance plot: {os.path.abspath(importance_path)}")
            plt.close()
            
        except Exception as e:
            print(f"\n⚠️ Could not create SHAP plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Return reduced datasets
    X_train_reduced = X_train[top_features]
    X_test_reduced = X_test[top_features]
    
    print(f"\n✅ Feature Reduction Complete")
    print(f"   Original: {len(X_test.columns)} features")
    print(f"   Reduced:  {len(top_features)} features ({len(top_features)/len(X_test.columns)*100:.1f}%)")
    print(f"   Coverage: {coverage:.1f}% of total SHAP importance\n")
    
    return top_features, shap_values, X_test_reduced


def retrain_with_selected_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_features: list,
    params: dict
) -> tuple:
    """
    Retrain model with only selected features.

    Returns:
        (retrained_model, test_metrics)
    """
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from xgboost import XGBClassifier

    print(f"\n🔄 Retraining model with {len(top_features)} selected features...")

    # Subset data
    X_train_reduced = X_train[top_features]
    X_test_reduced = X_test[top_features]

    # Retrain
    model_v2 = XGBClassifier(**params)
    model_v2.fit(X_train_reduced, y_train)

    # Evaluate
    y_pred_proba_v2 = model_v2.predict_proba(X_test_reduced)[:, 1]
    y_pred_v2 = (y_pred_proba_v2 > 0.5).astype(int)

    metrics_v2 = {
        "accuracy": accuracy_score(y_test, y_pred_v2),
        "logloss": log_loss(y_test, y_pred_proba_v2),
        "roc_auc": roc_auc_score(y_test, y_pred_proba_v2),
    }

    print(f"✅ Retrained Model Performance:")
    print(f"   Accuracy: {metrics_v2['accuracy']:.4f}")
    print(f"   Log Loss: {metrics_v2['logloss']:.4f}")
    print(f"   ROC-AUC:  {metrics_v2['roc_auc']:.4f}")

    return model_v2, metrics_v2
