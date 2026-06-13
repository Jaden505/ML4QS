"""Train XGBoost and Random Forest classifiers on engineered features."""

import json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS = {}


def prepare_data(feature_df, labels, participant_array, train_participants=None):
    """Split by participant into train/test. Returns X_train, X_test, y_train, y_test, label_encoder."""
    participants = sorted(np.unique(participant_array))
    if train_participants is None: train_participants = participants[:-1]
    test_participants = [p for p in participants if p not in train_participants]
    le = LabelEncoder()
    y_all = le.fit_transform(labels)
    train_mask = np.isin(participant_array, train_participants)
    X_train = feature_df.values[train_mask].astype(np.float32)
    X_test = feature_df.values[~train_mask].astype(np.float32)
    y_train = y_all[train_mask]
    y_test = y_all[~train_mask]
    print(f"Train ({train_participants}): {len(X_train)} samples  Test ({test_participants}): {len(X_test)} samples")
    print(f"Classes: {list(le.classes_)}")
    return X_train, X_test, y_train, y_test, le


def train_xgboost(X_train, y_train, search_iter=20, random_state=42):
    """XGBoost with randomised hyperparameter search."""
    import xgboost as xgb
    search = RandomizedSearchCV(xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                                                   random_state=random_state, n_jobs=-1, verbosity=0),
                                {"n_estimators": [100, 200, 300, 500], "max_depth": [3, 5, 7, 10],
                                 "learning_rate": [0.01, 0.05, 0.1, 0.2], "subsample": [0.6, 0.8, 1.0],
                                 "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5], "gamma": [0, 0.1, 0.2]},
                                n_iter=search_iter, cv=3, scoring="f1_weighted", verbose=0, random_state=random_state, n_jobs=-1)
    print("Training XGBoost...")
    search.fit(X_train, y_train)
    print(f"  Best CV F1: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


def train_random_forest(X_train, y_train, search_iter=20, random_state=42):
    """Random Forest with randomised hyperparameter search."""
    search = RandomizedSearchCV(RandomForestClassifier(random_state=random_state, n_jobs=-1),
                                {"n_estimators": [100, 200, 300, 500], "max_depth": [None, 10, 20, 30],
                                 "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
                                 "max_features": ["sqrt", "log2", None], "bootstrap": [True, False]},
                                n_iter=search_iter, cv=3, scoring="f1_weighted", verbose=0, random_state=random_state, n_jobs=-1)
    print("Training Random Forest...")
    search.fit(X_train, y_train)
    print(f"  Best CV F1: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


def train_and_save(feature_df, labels, participant_array, train_participants=None, output_dir=None,
                   search_iter=20, random_state=42, split_mode="participant"):
    """Full pipeline: split, train XGBoost + RF, save models + test data to disk."""
    if output_dir is None: output_dir = MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if split_mode == "random":
        from sklearn.model_selection import train_test_split as tts
        le = LabelEncoder()
        X_train, X_test, y_train, y_test = tts(feature_df.values.astype(np.float32), le.fit_transform(labels),
                                                test_size=0.2, random_state=random_state, stratify=labels)
        train_parts_str, test_parts_str = "random_80pct", "random_20pct"
        print(f"Random split — train: {len(X_train)}, test: {len(X_test)}  Classes: {list(le.classes_)}")
    else:
        X_train, X_test, y_train, y_test, le = prepare_data(feature_df, labels, participant_array, train_participants)
        train_parts_str, test_parts_str = str(train_participants), str([p for p in sorted(np.unique(participant_array)) if p not in (train_participants or [])])

    xgb_model, xgb_params = train_xgboost(X_train, y_train, search_iter, random_state)
    rf_model, rf_params = train_random_forest(X_train, y_train, search_iter // 2, random_state)

    for name, model in [("xgboost", xgb_model), ("random_forest", rf_model)]:
        with open(output_dir / f"{name}_model.pkl", "wb") as f:
            pickle.dump({"model": model, "label_encoder": le, "params": xgb_params if name == "xgboost" else rf_params}, f)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    with open(output_dir / "training_results.json", "w") as f:
        json.dump({"split_mode": split_mode, "train_participants": train_parts_str, "test_participants": test_parts_str,
                    "classes": list(le.classes_), "X_train_shape": list(X_train.shape), "X_test_shape": list(X_test.shape),
                    "xgboost_params": xgb_params, "random_forest_params": rf_params}, f, indent=2)
    print(f"  ✓ Saved models + test data to {output_dir}")

    MODELS.update({"xgboost": xgb_model, "random_forest": rf_model, "label_encoder": le, "X_test": X_test, "y_test": y_test})


def get_trained_models():
    """Return last trained models (in-memory)."""
    return MODELS


if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.windowing import create_windows
    from src.features.features import extract_features, add_magnitude_features
    df = load_data()
    windows, labels, meta = create_windows(df)
    feat_df, _ = extract_features(add_magnitude_features(windows), fs=100.0)
    print(f"Features: {feat_df.shape}")
    train_and_save(feat_df, meta["exercises"], meta["participants"], train_participants=["P1", "P2", "P3"], search_iter=20)
