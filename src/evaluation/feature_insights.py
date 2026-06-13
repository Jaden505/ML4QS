"""Feature importance analysis with SHAP and Random Forest importances."""

import pickle, json, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.0)


def extract_rf_importance(feature_names, model_dir=None, top_n=20):
    """Extract and rank top-N feature importances from Random Forest."""
    if model_dir is None: model_dir = MODEL_DIR
    with open(model_dir / "random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)["model"]
        
    imp = model.feature_importances_ if hasattr(model, "feature_importances_") else model.best_estimator_.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False).head(top_n)
    return df


def plot_rf_feature_importance(importance_df, save_path=None, top_n=20):
    """Horizontal bar chart of feature importances."""
    if save_path is None: save_path = FIGURE_DIR / "rf_feature_importance.png"
    data = importance_df.head(top_n).iloc[::-1]
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(range(len(data)), data["importance"].values, color="steelblue")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["feature"].values, fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Top Feature Importances")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    return str(save_path)


def plot_shap_summary(model, X_background, X_explain, feature_names, class_names, save_path=None, n_samples=100):
    """SHAP beeswarm plot for multi-class XGBoost."""
    if save_path is None: save_path = FIGURE_DIR / "shap_summary.png"
    
    X_back = X_background[np.random.choice(len(X_background), min(n_samples, len(X_background)), replace=False)]
    X_explain_subset = X_explain[:min(n_samples, len(X_explain))]
    explainer = shap.TreeExplainer(model, X_back, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_explain_subset)
    
    if isinstance(shap_values, list):
        fig, axes = plt.subplots(1, len(shap_values), figsize=(6 * len(shap_values), 5))
        if len(shap_values) == 1: axes = [axes]
        for i, (sv, cls_name) in enumerate(zip(shap_values, class_names)):
            shap.summary_plot(sv, X_explain_subset, feature_names=feature_names, show=False, ax=axes[i])
            axes[i].set_title(f"SHAP — {cls_name}")
        fig.tight_layout()
    else:
        shap.summary_plot(shap_values, X_explain_subset, feature_names=feature_names, show=False)
        fig = plt.gcf()
        
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(save_path)


def run_feature_insights(feature_names, X_test, y_test, class_names, model_dir=None, figure_dir=None, top_n=20):
    """Full feature importance analysis: RF importance + SHAP, prints top features + sensor group analysis."""
    if model_dir is None: model_dir = MODEL_DIR
    if figure_dir is None: figure_dir = FIGURE_DIR

    rf_df = extract_rf_importance(feature_names, model_dir, top_n=50)
    plot_rf_feature_importance(rf_df, figure_dir / "rf_feature_importance.png", top_n)

    with open(model_dir / "xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)["model"]
    plot_shap_summary(xgb_model, X_test, X_test, feature_names, class_names, figure_dir / "shap_summary.png", min(100, len(X_test)))

    print(f"\n{'='*60}\n  TOP {min(10, top_n)} FEATURES (Random Forest)\n{'='*60}")
    print(f"  {'Rank':<6s} {'Feature':<40s} {'Importance':>12s}")
    for rank, (_, row) in enumerate(rf_df.head(10).iterrows(), 1):
        print(f"  {rank:<6d} {row['feature']:<40s} {row['importance']:>12.4f}")

    sensors = {"accel": "accel", "gyro": "gyro", "lin_accel": "lin_accel"}
    print(f"\n  SENSOR GROUP IMPORTANCE (aggregated from top-50)")
    for s_name, prefix in sensors.items():
        mask = rf_df["feature"].str.contains(prefix, na=False)
        print(f"  {s_name:<15s} total_importance={rf_df.loc[mask, 'importance'].sum():.4f}  (features: {mask.sum()})")

    return {"rf_importance_top": rf_df.head(top_n).to_dict("records"), "rf_plot_path": str(figure_dir / "rf_feature_importance.png"),
            "shap_plot_path": str(figure_dir / "shap_summary.png")}


if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.windowing import create_windows
    from src.features.features import extract_features, add_magnitude_features
    
    df = load_data()
    windows, labels, meta = create_windows(df)
    feat_df, feat_names = extract_features(add_magnitude_features(windows), fs=100.0)
    X_test = np.load(MODEL_DIR / "X_test.npy")
    y_test = np.load(MODEL_DIR / "y_test.npy")
    with open(MODEL_DIR / "training_results.json") as f:
        class_names = json.load(f)["classes"]
    run_feature_insights(feat_names, X_test, y_test, class_names)
