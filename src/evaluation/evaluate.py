"""Evaluation metrics, confusion matrix, and per-class breakdown."""

import pickle, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)


def load_models(model_dir=None):
    """Load trained models and test data from disk."""
    if model_dir is None: model_dir = MODEL_DIR
    models = {}
    for name in ["xgboost", "random_forest"]:
        with open(model_dir / f"{name}_model.pkl", "rb") as f:
            models[name] = pickle.load(f)["model"]
    if not (model_dir / "X_test.npy").exists():
        raise FileNotFoundError("Run train_classical.py first.")
    return models, np.load(model_dir / "X_test.npy"), np.load(model_dir / "y_test.npy")


def evaluate_models(models, X_test, y_test, class_names=None):
    """Print per-class metrics for each model. Returns dict of results."""
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        print(f"\n{'='*60}\n  {name.upper()}\n{'='*60}")
        print(f"  Accuracy: {accuracy_score(y_test, preds):.4f}  F1 (wgt): {f1_score(y_test, preds, average='weighted'):.4f}")
        print(f"  {'Class':<20s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Supp':>8s}")
        print(f"  {'-'*52}")
        report = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)
        
        for cls in (class_names or []):
            if cls in report:
                r = report[cls]
                print(f"  {cls:<20s} {r['precision']:>8.3f} {r['recall']:>8.3f} {r['f1-score']:>8.3f} {r['support']:>8.0f}")
        results[name] = {"accuracy": accuracy_score(y_test, preds), "f1_weighted": f1_score(y_test, preds, average="weighted"),
                         "f1_macro": f1_score(y_test, preds, average="macro")}
        
    return results


def plot_confusion_matrices(models, X_test, y_test, class_names=None, save_dir=None):
    """Generate and save confusion matrix plots. Returns dict of path per model."""
    if save_dir is None: save_dir = FIGURE_DIR
    paths = {}
    
    for name, model in models.items():
        cm = confusion_matrix(y_test, model.predict(X_test))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"Confusion Matrix — {name.upper()}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        path = save_dir / f"confusion_matrix_{name}.png"
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        paths[name] = str(path)
        
    return paths


def run_evaluation(model_dir=None, figure_dir=None):
    """Run full evaluation: load → metrics → confusion matrices."""
    if model_dir is None: model_dir = MODEL_DIR
    if figure_dir is None: figure_dir = FIGURE_DIR
    
    with open(model_dir / "xgboost_model.pkl", "rb") as f:
        class_names = list(pickle.load(f)["label_encoder"].classes_)
        
    models, X_test, y_test = load_models(model_dir)
    print(f"Test samples: {len(y_test)}  Classes: {class_names}")
    results = evaluate_models(models, X_test, y_test, class_names)
    cm_paths = plot_confusion_matrices(models, X_test, y_test, class_names, figure_dir)
    
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  {'Model':<20s} {'Accuracy':>10s} {'F1 (wgt)':>10s} {'F1 (macro)':>10s}")
    for name, m in results.items():
        print(f"  {name:<20s} {m['accuracy']:>10.4f} {m['f1_weighted']:>10.4f} {m['f1_macro']:>10.4f}")
    return {"metrics": results, "confusion_matrix_paths": cm_paths}


if __name__ == "__main__":
    run_evaluation()
