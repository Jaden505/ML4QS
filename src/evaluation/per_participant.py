"""Leave-one-participant-out per-class F1 bar chart."""

from pathlib import Path
import matplotlib

from src.features.features import add_magnitude_features
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_per_participant(feature_df, labels, participant_array, search_iter=5):
    """Leave-one-participant-out: trains on N-1 participants, tests on held-out, plots per-exercise F1."""
    from src.train_classical import train_xgboost, train_random_forest

    participants = sorted(np.unique(participant_array))
    class_names = sorted(np.unique(labels))
    le = LabelEncoder()
    y_all = le.fit_transform(labels)
    X_all = feature_df.values.astype(np.float32)

    rows = []
    for test_p in participants:
        mask = participant_array != test_p
        X_tr, X_te = X_all[mask], X_all[~mask]
        y_tr, y_te = y_all[mask], y_all[~mask]
        print(f"  Test: {test_p}  (train: {len(X_tr)}, test: {len(X_te)})")

        for model_name, model_fn, iters in [("XGBoost", train_xgboost, search_iter),
                                            ("Random Forest", train_random_forest, search_iter // 2)]:
            model, _ = model_fn(X_tr, y_tr, iters)
            report = classification_report(y_te, model.predict(X_te), output_dict=True, zero_division=0)
            for cls in class_names:
                key = str(le.transform([cls])[0])
                if key in report:
                    rows.append({"participant": test_p, "exercise": cls, "model": model_name,
                                 "f1": report[key]["f1-score"], "support": report[key]["support"]})

    df = pd.DataFrame(rows)
    return df
    
def plot_per_participant_f1(df, figure_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.barplot(data=df, x="exercise", y="f1", hue="model", ax=axes[0], ci=None)
    axes[0].set_title("Per-Participant F1 Score by Exercise")
    axes[0].set_xlabel("Exercise")
    axes[0].set_ylabel("F1 Score")
    axes[0].legend(title="Model")
    sns.barplot(data=df, x="participant", y="f1", hue="model", ax=axes[1], ci=None)
    axes[1].set_title("Per-Participant F1 Score by Participant")
    axes[1].set_xlabel("Participant")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend(title="Model")
    fig.tight_layout()
    if figure_dir is not None:
        path = figure_dir / "per_participant_f1.png"
        fig.savefig(path, dpi=150)
        print(f"Saved per-participant F1 plot to: {path}")
    plt.close(fig)
    
if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.features.features import extract_features
    from src.data.windowing import create_windows

    df = load_data()
    windows, labels, meta = create_windows(df)
    feat_df, _ = extract_features(add_magnitude_features(windows), fs=100.0)
    print(f"Features: {feat_df.shape}")
    
    eval_df = evaluate_per_participant(feat_df, meta["exercises"], meta["participants"], search_iter=10)
    plot_per_participant_f1(eval_df, FIGURE_DIR)
