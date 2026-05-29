# patched_feature_selection.py
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

try:
    from predictive_model.model_xgb import MODEL_DIR
except ImportError:
    try:
        from config import MODEL_DIR
    except ImportError:
        MODEL_DIR = "models"


def get_optimal_feature_count(n_samples: int, n_features: int, mode: str = "daily") -> int:
    max_features_by_samples = n_samples // 10
    min_features_pct = int(n_features * 0.30)
    max_features_pct = int(n_features * 0.50)

    if mode in ("intraday", "intraday_mr", "intraday_mom"):
        lower_bound = 30
        upper_bound = 60
    else:
        lower_bound = 25
        upper_bound = 50

    optimal = min(max_features_pct, max_features_by_samples)
    optimal = max(optimal, min_features_pct)
    optimal = max(optimal, lower_bound)
    optimal = min(optimal, upper_bound)
    optimal = min(optimal, n_features)
    return optimal


def prune_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.97,
    preserve: list | None = None,
):
    """
    Remove one feature from highly correlated pairs.
    Preserve list can pin features that should survive pruning.
    """
    preserve = set(preserve or [])
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = []
    for col in upper.columns:
        hits = upper.index[upper[col] > threshold].tolist()
        if not hits:
            continue
        if col in preserve:
            continue
        if any(h in preserve for h in hits):
            to_drop.append(col)
        else:
            to_drop.append(col)

    X_pruned = X.drop(columns=sorted(set(to_drop)), errors="ignore")
    return X_pruned, sorted(set(to_drop))


def select_features_with_shap(
    model,
    X_train: pd.DataFrame,
    X_cal: pd.DataFrame,
    top_n: int = None,
    plot: bool = True,
    symbol: str = "UNKNOWN",
    mode: str = "daily",
    correlation_prune_threshold: float = 0.97,
) -> tuple:
    """
    Use SHAP to select most important features.
    IMPORTANT: selection is done on calibration data, not final test data.

    Returns:
    (top_features_list, shap_values_cal, X_train_reduced, X_cal_reduced, metadata)
    """
    n_samples, n_features = X_train.shape

    if top_n is None:
        top_n = get_optimal_feature_count(n_samples, n_features, mode)
        print(f"\n[SHAP] Feature Selection (AUTO)")
        print(f" Total features: {n_features}")
        print(f" Training samples: {n_samples}")
        print(f" Mode: {mode}")
        print(f" Selected target: {top_n} features ({top_n / n_features * 100:.1f}%)")
    else:
        print(f"\n[SHAP] Feature Selection (MANUAL)")
        print(f" Total features: {n_features}")
        print(f" Selecting top {top_n} features")

    if top_n > n_features:
        top_n = n_features
    elif top_n < 10:
        top_n = max(top_n, 10)

    model_dir = MODEL_DIR if MODEL_DIR else "models"
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n[PRUNE] Correlation pruning before SHAP...")
    preserve = [c for c in X_train.columns if c.startswith("spy_") or c.startswith("is_")]
    X_train_pruned, dropped_corr = prune_correlated_features(
        X_train,
        threshold=correlation_prune_threshold,
        preserve=preserve,
    )
    X_cal_pruned = X_cal[X_train_pruned.columns].copy()

    print(f"[PRUNE] Dropped {len(dropped_corr)} correlated features")
    print(f"[PRUNE] Remaining features: {X_train_pruned.shape[1]}")

    print(f"\n[SHAP] Creating TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    print(f"[SHAP] Computing SHAP values on {len(X_cal_pruned)} calibration samples...")
    shap_values = explainer.shap_values(X_cal_pruned)

    feature_importance = pd.DataFrame({
        "feature": X_cal_pruned.columns,
        "shap_importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("shap_importance", ascending=False)

    top_features = feature_importance.head(top_n)["feature"].tolist()

    total_importance = feature_importance["shap_importance"].sum()
    top_n_importance = feature_importance.head(top_n)["shap_importance"].sum()
    coverage = (top_n_importance / total_importance * 100) if total_importance > 1e-12 else 0.0

    print(f"\n[SHAP] Selected Top {top_n} Features (covering {coverage:.1f}% of total importance)")
    print("=" * 70)
    print(f"{'Rank':<6} {'Feature':<35} {'SHAP Importance':<15} {'Cumulative %'}")
    print("-" * 70)

    cumulative_importance = 0.0
    for rank, (_, row) in enumerate(feature_importance.head(top_n).iterrows(), start=1):
        cumulative_importance += row["shap_importance"]
        cumulative_pct = (cumulative_importance / total_importance * 100) if total_importance > 1e-12 else 0.0
        print(f"{rank:<6} {row['feature']:<35} {row['shap_importance']:<15.6f} {cumulative_pct:>6.1f}%")
    print("=" * 70)

    if plot:
        try:
            summary_filename = f"{symbol}_{mode}_shap_summary_top{top_n}.png"
            importance_filename = f"{symbol}_{mode}_shap_importance_top{top_n}.png"
            summary_path = os.path.join(model_dir, summary_filename)
            importance_path = os.path.join(model_dir, importance_filename)

            print(f"\n[SHAP] Generating plots...")
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_cal_pruned, max_display=min(top_n, 30), show=False)
            plt.title(f"SHAP Feature Summary - {symbol} {mode.upper()} (Top {top_n})", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(summary_path, dpi=150, bbox_inches="tight")
            plt.close()

            fig, ax = plt.subplots(figsize=(12, 10))
            display_count = min(30, top_n)
            top_display = feature_importance.head(display_count)
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_display)))
            bars = ax.barh(range(len(top_display)), top_display["shap_importance"], color=colors)
            ax.set_yticks(range(len(top_display)))
            ax.set_yticklabels(top_display["feature"], fontsize=9)
            ax.set_xlabel("Mean |SHAP value|", fontsize=11)
            ax.set_title(f"Feature Importance - {symbol} {mode.upper()} (Top {display_count})", fontsize=14, fontweight="bold")
            ax.invert_yaxis()
            for i, (bar, val) in enumerate(zip(bars, top_display["shap_importance"])):
                ax.text(val, i, f" {val:.4f}", va="center", fontsize=8)
            plt.tight_layout()
            plt.savefig(importance_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"[SHAP] Could not create plots: {e}")

    X_train_reduced = X_train[top_features].copy()
    X_cal_reduced = X_cal[top_features].copy()

    metadata = {
        "selection_window": "calibration",
        "top_n": len(top_features),
        "coverage_pct": float(coverage),
        "dropped_correlated_features": dropped_corr,
        "selected_features": top_features,
        "feature_importance": feature_importance.to_dict(orient="records"),
    }

    print(f"\n[SHAP] Feature Reduction Complete")
    print(f" Original: {len(X_train.columns)} features")
    print(f" Reduced: {len(top_features)} features ({len(top_features) / len(X_train.columns) * 100:.1f}%)")
    print(f" Coverage: {coverage:.1f}% of total SHAP importance\n")

    return top_features, shap_values, X_train_reduced, X_cal_reduced, metadata


def retrain_with_selected_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    top_features: list,
    params: dict,
) -> tuple:
    """
    Retrain model with only selected features.
    Returns:
    (retrained_model, eval_metrics)
    """
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from xgboost import XGBClassifier

    print(f"\n[RETRAIN] Retraining model with {len(top_features)} selected features...")

    X_train_reduced = X_train[top_features]
    X_eval_reduced = X_eval[top_features]

    model_v2 = XGBClassifier(**params)
    model_v2.fit(X_train_reduced, y_train)

    y_pred_proba_v2 = model_v2.predict_proba(X_eval_reduced)[:, 1]
    y_pred_v2 = (y_pred_proba_v2 > 0.5).astype(int)

    metrics_v2 = {
        "accuracy": accuracy_score(y_eval, y_pred_v2),
        "logloss": log_loss(y_eval, y_pred_proba_v2),
        "roc_auc": roc_auc_score(y_eval, y_pred_proba_v2) if len(np.unique(y_eval)) >= 2 else None,
    }

    print(f"[RETRAIN] Accuracy: {metrics_v2['accuracy']:.4f}")
    print(f"[RETRAIN] Log Loss: {metrics_v2['logloss']:.4f}")
    if metrics_v2["roc_auc"] is not None:
        print(f"[RETRAIN] ROC-AUC: {metrics_v2['roc_auc']:.4f}")

    return model_v2, metrics_v2
