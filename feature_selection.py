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

def select_features_with_shap(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    top_n: int = 30,
    plot: bool = True,
    symbol: str = "UNKNOWN",  # ✅ NEW: Symbol for filename
    mode: str = "model"        # ✅ NEW: Mode (daily/intraday_mr/intraday_mom)
) -> tuple:
    """
    Use SHAP to select most important features.

    Args:
        model: Trained XGBoost model
        X_train: Training features
        X_test: Test features
        top_n: Number of top features to keep
        plot: Whether to show SHAP plots
        symbol: Stock symbol (e.g., "NVDA") for filename
        mode: Model type (e.g., "daily", "intraday_mr", "intraday_mom")

    Returns:
        (top_features_list, shap_values_test, X_test_reduced)
    """
    print(f"\n🔍 Running SHAP feature selection (top {top_n} features)...")

    # ✅ Ensure models/ directory exists
    model_dir = MODEL_DIR if MODEL_DIR else "models"
    os.makedirs(model_dir, exist_ok=True)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values on test set (smaller, faster)
    shap_values = explainer.shap_values(X_test)

    # Get feature importance
    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "shap_importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("shap_importance", ascending=False)

    # Select top N features
    top_features = feature_importance.head(top_n)["feature"].tolist()

    print(f"\n✅ Top {top_n} Features by SHAP Importance:")
    print("-" * 60)
    for i, row in feature_importance.head(top_n).iterrows():
        print(f"{row['feature']:30} {row['shap_importance']:.6f}")
    print("-" * 60)

    # Optionally plot
    if plot:
        try:
            # ✅ FIXED: Descriptive filenames in models/ directory
            summary_filename = f"{symbol}_{mode}_shap_summary_top{top_n}.png"
            importance_filename = f"{symbol}_{mode}_shap_importance_top{top_n}.png"

            summary_path = os.path.join(model_dir, summary_filename)
            importance_path = os.path.join(model_dir, importance_filename)

            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X_test,
                max_display=top_n,
                show=False
            )
            plt.title(f"Top {top_n} Features - SHAP Summary ({symbol} {mode.upper()})")
            plt.tight_layout()
            plt.savefig(summary_path, dpi=150, bbox_inches="tight")
            print(f"📊 SHAP summary saved: {os.path.abspath(summary_path)}")
            plt.close()

            # Bar plot of importance
            fig, ax = plt.subplots(figsize=(10, 8))
            top_20 = feature_importance.head(20)
            ax.barh(range(len(top_20)), top_20["shap_importance"])
            ax.set_yticks(range(len(top_20)))
            ax.set_yticklabels(top_20["feature"])
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"Feature Importance - SHAP ({symbol} {mode.upper()})")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(importance_path, dpi=150, bbox_inches="tight")
            print(f"📊 Importance plot saved: {os.path.abspath(importance_path)}")
            plt.close()

        except Exception as e:
            print(f"[WARN] Could not create SHAP plots: {e}")

    # Return reduced datasets
    X_train_reduced = X_train[top_features]
    X_test_reduced = X_test[top_features]

    print(f"\n✅ Feature reduction: {len(X_test.columns)} → {len(top_features)} features")

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
