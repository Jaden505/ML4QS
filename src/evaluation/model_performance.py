"""Evaluation metrics, confusion matrix, per-participant F1 breakdown."""

import pickle, warnings
from pathlib import Path
import matplotlib
from sklearn.model_selection import train_test_split
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

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
    """Generate and save confusion matrix plots (only for classes present in test set)."""
    if save_dir is None: save_dir = FIGURE_DIR
    paths = {}
    present_classes = sorted(np.unique(y_test))
    present_names = [class_names[i] for i in present_classes] if class_names else present_classes
    
    for name, model in models.items():
        cm = confusion_matrix(y_test, model.predict(X_test), labels=present_classes)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=present_names, yticklabels=present_names, ax=ax)
        ax.set_title(f"Confusion Matrix — {name.upper()}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        path = save_dir / f"confusion_matrix_{name}.png"
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        paths[name] = str(path)
        
    return paths

if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.windowing import create_windows
    from src.features.features import extract_features, add_magnitude_features
    from src.train_classical import train_xgboost, train_random_forest
    from src.features.augmentation import augment_dataset
    from src.evaluation.compare_groups import evaluate_per_participant
    
    df = load_data()
    windows, labels, meta = create_windows(df)

    # Track original size before augmentation so we can replicate metadata
    n_orig = len(windows)
    windows, labels = augment_dataset(windows, labels, augment_factor=2, seed=42)
    factor = len(windows) // n_orig  # e.g. 3 (original + 2 augmented copies)

    # Augment metadata arrays to stay aligned with the multiplied windows
    for key in ["participants", "exercises"]:
        meta[key] = np.concatenate([meta[key]] * factor)

    feat_df, feat_names = extract_features(add_magnitude_features(windows), fs=100.0)

    # Per-participant evaluation with augmented data
    per_participant_df = evaluate_per_participant(feat_df, meta["exercises"],
                                                   meta["participants"], search_iter=5)

    # Random-split evaluation with augmented data
    X_train, X_test, y_train, y_test = train_test_split(
        feat_df.values.astype(np.float32), labels, test_size=0.2,
        random_state=42, stratify=labels)

    xgb_model, _ = train_xgboost(X_train, y_train, search_iter=10)
    models = {"xgboost": xgb_model}
    unique_class_names = sorted(np.unique(meta["exercises"]))
    evaluate_models(models, X_test, y_test, class_names=unique_class_names)
    plot_confusion_matrices(models, X_test, y_test, class_names=unique_class_names)
    