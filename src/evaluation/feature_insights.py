"""Feature importance analysis with SHAP and Random Forest importances."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)


# ---------------------------------------------------------------------------
# RF feature importance
# ---------------------------------------------------------------------------


def extract_rf_importance(
    feature_names: List[str],
    model_dir: Optional[Path] = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract and rank feature importances from trained Random Forest.

    Returns a DataFrame sorted by importance (descending).
    """
    if model_dir is None:
        model_dir = MODEL_DIR

    rf_path = model_dir / "random_forest_model.pkl"
    if not rf_path.exists():
        raise FileNotFoundError(f"Random Forest model not found: {rf_path}")

    with open(rf_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "best_estimator_"):
        importances = model.best_estimator_.feature_importances_
    else:
        raise AttributeError("Model does not have feature_importances_")

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def plot_rf_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Random Forest — Top Feature Importances",
    save_path: Optional[Path] = None,
    top_n: int = 20,
) -> str:
    """Plot horizontal bar chart of feature importances."""
    if save_path is None:
        save_path = FIGURE_DIR / "rf_feature_importance.png"

    data = importance_df.head(top_n).iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(range(len(data)), data["importance"].values, color="steelblue")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["feature"].values, fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()  # highest at top

    # Annotate values
    for bar, val in zip(bars, data["importance"].values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ RF feature importance plot saved: {save_path}")
    return str(save_path)


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    n_samples: int = 100,
) -> Any:
    """Compute SHAP values using TreeExplainer (works for both XGBoost and RF).

    Parameters
    ----------
    model : Any
        Trained tree-based model (XGBoost or Random Forest).
    X_background : np.ndarray
        Background dataset for TreeExplainer (small subset is fine).
    X_explain : np.ndarray
        Samples to explain.
    n_samples : int
        Number of background samples to use.

    Returns
    -------
    shap_values — output from TreeExplainer.shap_values().
    """
    import shap

    # Sample background
    if len(X_background) > n_samples:
        idx = np.random.choice(len(X_background), n_samples, replace=False)
        X_back = X_background[idx]
    else:
        X_back = X_background

    explainer = shap.TreeExplainer(model, X_back, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_explain[:n_samples] if len(X_explain) > n_samples else X_explain)
    return shap_values


def plot_shap_summary(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    save_path: Optional[Path] = None,
    n_samples: int = 100,
) -> str:
    """Generate SHAP summary (beeswarm) plot for a multi-class model."""
    import shap

    if save_path is None:
        save_path = FIGURE_DIR / "shap_summary.png"

    # SHAP for multi-class returns a list of arrays (one per class)
    shap_values = compute_shap_values(model, X_background, X_explain, n_samples)

    X_display = X_explain[:min(n_samples, len(X_explain))]

    if isinstance(shap_values, list):
        # Multi-class: plot one summary per class (or combined)
        n_classes = len(shap_values)
        fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
        if n_classes == 1:
            axes = [axes]

        for i, (sv, cls_name) in enumerate(zip(shap_values, class_names)):
            shap.summary_plot(
                sv,
                X_display,
                feature_names=feature_names,
                class_names=class_names,
                class_inds=i,
                show=False,
                ax=axes[i],
            )
            axes[i].set_title(f"SHAP — {cls_name}")
        fig.tight_layout()
    else:
        # Binary / single-output
        shap.summary_plot(
            shap_values, X_display, feature_names=feature_names, show=False
        )
        fig = plt.gcf()

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ SHAP summary plot saved: {save_path}")
    return str(save_path)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def print_top_features(
    importance_df: pd.DataFrame,
    shap_top: Optional[List[Tuple[str, float]]] = None,
    top_n: int = 10,
) -> None:
    """Print a human-readable summary of top features."""
    print(f"\n{'='*60}")
    print(f"  TOP {top_n} FEATURES (Random Forest)")
    print(f"{'='*60}")
    print(f"  {'Rank':<6s} {'Feature':<40s} {'Importance':>12s}")
    print(f"  {'-'*58}")
    for rank, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
        print(f"  {rank:<6d} {row['feature']:<40s} {row['importance']:>12.4f}")

    if shap_top:
        print(f"\n{'='*60}")
        print(f"  TOP {len(shap_top)} FEATURES (SHAP — XGBoost)")
        print(f"{'='*60}")
        print(f"  {'Rank':<6s} {'Feature':<40s} {'Mean |SHAP|':>12s}")
        print(f"  {'-'*58}")
        for rank, (feat, val) in enumerate(shap_top[:top_n], 1):
            print(f"  {rank:<6d} {feat:<40s} {val:>12.4f}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_feature_insights(
    feature_names: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    model_dir: Optional[Path] = None,
    figure_dir: Optional[Path] = None,
    top_n: int = 20,
) -> Dict[str, Any]:
    """Run the full feature importance analysis pipeline."""
    if model_dir is None:
        model_dir = MODEL_DIR
    if figure_dir is None:
        figure_dir = FIGURE_DIR

    # --- RF feature importance ---
    print("Extracting Random Forest feature importances...")
    rf_imp_df = extract_rf_importance(feature_names, model_dir, top_n=50)
    rf_plot_path = plot_rf_feature_importance(rf_imp_df, top_n=top_n, save_path=figure_dir / "rf_feature_importance.png")

    # --- SHAP (XGBoost) ---
    print("\nComputing SHAP values for XGBoost...")
    xgb_path = model_dir / "xgboost_model.pkl"
    with open(xgb_path, "rb") as f:
        xgb_data = pickle.load(f)
    xgb_model = xgb_data["model"]

    shap_path = plot_shap_summary(
        xgb_model,
        X_test, X_test,
        feature_names=feature_names,
        class_names=class_names,
        save_path=figure_dir / "shap_summary.png",
        n_samples=min(100, len(X_test)),
    )

    # --- SHAP top features (mean |SHAP| across classes) ---
    import shap
    shap_values = compute_shap_values(xgb_model, X_test, X_test, n_samples=min(100, len(X_test)))
    shap_top: Optional[List[Tuple[str, float]]] = None
    if isinstance(shap_values, list):
        # Multi-class — average absolute SHAP across all classes
        mean_abs_shap = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    # Convert to plain Python ints for indexing
    top_idx_list = [int(i) for i in top_indices]
    shap_top = [(feature_names[idx], float(mean_abs_shap[idx])) for idx in top_idx_list]

    # --- Print ---
    print_top_features(rf_imp_df, shap_top, top_n=10)

    # --- Additional insight: which sensors matter most? ---
    print(f"\n{'='*60}")
    print(f"  SENSOR GROUP IMPORTANCE (aggregated)")
    print(f"{'='*60}")
    sensors = {"accel": ["accel"], "gyro": ["gyro"], "lin_accel": ["lin_accel"]}
    for sensor_name, prefixes in sensors.items():
        mask = rf_imp_df["feature"].str.contains("|".join(prefixes), na=False)
        total_imp = rf_imp_df.loc[mask, "importance"].sum()
        count = mask.sum()
        print(f"  {sensor_name:<15s} total_importance={total_imp:.4f}  (top-50 features: {count})")

    return {
        "rf_importance_top": rf_imp_df.head(top_n).to_dict("records"),
        "shap_top_features": shap_top if shap_top else [],
        "rf_plot_path": rf_plot_path,
        "shap_plot_path": shap_path,
    }


if __name__ == "__main__":
    import json
    import pickle

    from src.data.load_data import load_data
    from src.data.windowing import create_windows
    from src.features.features import extract_features, add_magnitude_features

    print("Loading data and extracting features...")
    df = load_data()
    windows, labels, meta = create_windows(df)
    windows_w_mag = add_magnitude_features(windows)
    feat_df, feat_names = extract_features(windows_w_mag, fs=100.0)

    # Load from disk
    X_test = np.load(MODEL_DIR / "X_test.npy")
    y_test = np.load(MODEL_DIR / "y_test.npy")
    with open(MODEL_DIR / "training_results.json") as f:
        train_results = json.load(f)
    class_names = train_results["classes"]

    print(f"Test samples: {len(X_test)}, Features: {len(feat_names)}")
    print(f"Classes: {class_names}")

    run_feature_insights(
        feature_names=feat_names,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        top_n=20,
    )
