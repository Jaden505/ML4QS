"""Evaluation metrics, confusion matrix, per-participant F1 breakdown."""

import pickle, warnings
from pathlib import Path
import matplotlib
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
    from sklearn.preprocessing import LabelEncoder

    # Load and process data (same pipeline as train_classical.py)
    df = load_data()
    windows, labels_arr, meta = create_windows(df)
    feat_df, _ = extract_features(add_magnitude_features(windows), fs=100.0)
    print(f"Feature matrix: {feat_df.shape}")

    # Prepare labels, features, and participant array
    participants = sorted(np.unique(meta["participants"]))
    class_names = sorted(np.unique(meta["exercises"]))
    le = LabelEncoder()
    y_all = le.fit_transform(meta["exercises"])
    X_all = feat_df.values.astype(np.float32)
    p_array = meta["participants"]

    print(f"Participants: {participants}")
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(X_all)}\n")

    # Leave-one-participant-out evaluation
    all_preds: dict[str, list[int]] = {"xgboost": [], "random_forest": []}
    all_trues: dict[str, list[int]] = {"xgboost": [], "random_forest": []}

    for test_p in participants:
        mask = p_array != test_p
        X_tr, X_te = X_all[mask], X_all[~mask]
        y_tr, y_te = y_all[mask], y_all[~mask]
        print(f"{'='*60}")
        print(f"  Hold-out participant: {test_p}")
        print(f"  Train: {len(X_tr)} samples, Test: {len(X_te)} samples")
        print(f"{'='*60}")

        for name, fn, iters in [("xgboost", train_xgboost, 20),
                                 ("random_forest", train_random_forest, 10)]:
            model, _ = fn(X_tr, y_tr, iters)
            preds = model.predict(X_te)
            all_preds[name].extend(preds.tolist())
            all_trues[name].extend(y_te.tolist())

            report = classification_report(y_te, preds, target_names=class_names,
                                            output_dict=True, zero_division=0)
            print(f"\n  {name.upper()}:")
            print(f"  {'Class':<20s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Supp':>8s}")
            print(f"  {'-'*52}")
            for cls in class_names:
                r = report[cls]
                print(f"  {cls:<20s} {r['precision']:>8.3f} {r['recall']:>8.3f}"
                      f" {r['f1-score']:>8.3f} {r['support']:>8.0f}")
        print()

    # Aggregate results across all folds
    print(f"\n{'='*60}")
    print("  AGGREGATED RESULTS (all held-out folds combined)")
    print(f"{'='*60}")
    results = {}
    for name in ["xgboost", "random_forest"]:
        y_true = np.array(all_trues[name])
        y_pred = np.array(all_preds[name])
        print(f"\n  {name.upper()}:")
        print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}"
              f"  F1 (wgt): {f1_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"  {'Class':<20s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Supp':>8s}")
        print(f"  {'-'*52}")
        report = classification_report(y_true, y_pred, target_names=class_names,
                                        output_dict=True, zero_division=0)
        for cls in class_names:
            r = report[cls]
            print(f"  {cls:<20s} {r['precision']:>8.3f} {r['recall']:>8.3f}"
                  f" {r['f1-score']:>8.3f} {r['support']:>8.0f}")
        results[name] = {"accuracy": accuracy_score(y_true, y_pred),
                         "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
                         "f1_macro": f1_score(y_true, y_pred, average="macro")}

    # Confusion matrices from aggregated (participant-aware) predictions
    for name in ["xgboost", "random_forest"]:
        y_true = np.array(all_trues[name])
        y_pred = np.array(all_preds[name])
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"Confusion Matrix — {name.upper()} (participant-aware)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        path = FIGURE_DIR / f"confusion_matrix_{name}_participant_aware.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved participant-aware confusion matrix: {path}")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for model_name, metrics in results.items():
        print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f},"
              f" F1 (weighted)={metrics['f1_weighted']:.4f},"
              f" F1 (macro)={metrics['f1_macro']:.4f}")