"""Evaluation metrics, confusion matrix, and per-class breakdown."""

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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

matplotlib.use("Agg")  # non-interactive backend

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def load_models(
    model_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Load trained models and test data from disk."""
    if model_dir is None:
        model_dir = MODEL_DIR

    models: Dict[str, Any] = {}

    for name in ["xgboost", "random_forest"]:
        path = model_dir / f"{name}_model.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        models[name] = data["model"]

    # Load test data from saved .npy files
    X_test_path = model_dir / "X_test.npy"
    y_test_path = model_dir / "y_test.npy"
    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {model_dir}/. Run train_classical.py first."
        )
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    return models, X_test, y_test


def evaluate_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class and overall metrics for each model.

    Returns
    -------
    results : dict
        ``{model_name: {"accuracy": ..., "f1_weighted": ..., "per_class": {...}}}``
    """
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        f1_w = f1_score(y_test, preds, average="weighted")
        f1_m = f1_score(y_test, preds, average="macro")
        prec_w = precision_score(y_test, preds, average="weighted")
        rec_w = recall_score(y_test, preds, average="weighted")

        report = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)

        results[name] = {
            "accuracy": acc,
            "f1_weighted": f1_w,
            "f1_macro": f1_m,
            "precision_weighted": prec_w,
            "recall_weighted": rec_w,
            "classification_report": report,
        }

        print(f"\n{'='*60}")
        print(f"  {name.upper()}")
        print(f"{'='*60}")
        print(f"  Accuracy:          {acc:.4f}")
        print(f"  F1 (weighted):     {f1_w:.4f}")
        print(f"  F1 (macro):        {f1_m:.4f}")
        print(f"  Precision (wgt):   {prec_w:.4f}")
        print(f"  Recall (wgt):      {rec_w:.4f}")
        print(f"\n  Per-class breakdown:")
        print(f"  {'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
        print(f"  {'-'*60}")
        for cls in class_names if class_names else sorted(report.keys() - {"accuracy", "macro avg", "weighted avg"}):
            if cls in report:
                r = report[cls]
                print(f"  {cls:<20s} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1-score']:>10.3f} {r['support']:>10.0f}")

    return results


def plot_confusion_matrix(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Generate and save confusion matrix plots for each model."""
    if save_dir is None:
        save_dir = FIGURE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, str] = {}

    for name, model in models.items():
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {name.upper()}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        path = save_dir / f"confusion_matrix_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths[name] = str(path)
        print(f"  ✓ Confusion matrix saved: {path}")

    return saved_paths


def run_evaluation(
    model_dir: Optional[Path] = None,
    figure_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline."""
    if model_dir is None:
        model_dir = MODEL_DIR
    if figure_dir is None:
        figure_dir = FIGURE_DIR

    print("Loading models and test data...")
    models, X_test, y_test = load_models(model_dir)
    le_path = model_dir / "xgboost_model.pkl"
    with open(le_path, "rb") as f:
        le = pickle.load(f)["label_encoder"]
    class_names = list(le.classes_)

    print(f"Test samples: {len(y_test)}")
    print(f"Classes: {class_names}")

    results = evaluate_models(models, X_test, y_test, class_names)
    cm_paths = plot_confusion_matrix(models, X_test, y_test, class_names, figure_dir)

    # Print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<20s} {'Accuracy':>10s} {'F1 (wgt)':>10s} {'F1 (macro)':>10s}")
    print(f"  {'-'*50}")
    for name, metrics in results.items():
        print(f"  {name:<20s} {metrics['accuracy']:>10.4f} {metrics['f1_weighted']:>10.4f} {metrics['f1_macro']:>10.4f}")

    return {
        "metrics": results,
        "confusion_matrix_paths": cm_paths,
    }


if __name__ == "__main__":
    run_evaluation()
