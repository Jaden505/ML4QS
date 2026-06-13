"""Train XGBoost and Random Forest classifiers on engineered features."""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_data(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    participant_array: np.ndarray,
    train_participants: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Split data into train/test by participant.

    Parameters
    ----------
    feature_df : pd.DataFrame, shape (N, num_features)
    labels : np.ndarray, shape (N,)
    participant_array : np.ndarray, shape (N,)
    train_participants : list of str, optional
        Participants for training. Defaults to all except the last.

    Returns
    -------
    X_train, X_test, y_train, y_test, label_encoder
    """
    participants = sorted(np.unique(participant_array))
    if train_participants is None:
        train_participants = participants[:-1]  # leave last participant out

    test_participants = [p for p in participants if p not in train_participants]

    train_mask = np.isin(participant_array, train_participants)
    test_mask = np.isin(participant_array, test_participants)

    le = LabelEncoder()
    y_all = le.fit_transform(labels)

    X_train = feature_df.values[train_mask].astype(np.float32)
    X_test = feature_df.values[test_mask].astype(np.float32)
    y_train = y_all[train_mask]
    y_test = y_all[test_mask]

    print(f"Train participants: {train_participants} → {len(X_train)} samples")
    print(f"Test participants:  {test_participants} → {len(X_test)} samples")
    print(f"Classes: {list(le.classes_)}")

    return X_train, X_test, y_train, y_test, le


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_iter: int = 30,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """Train XGBoost with randomised hyperparameter search.

    Returns (best_estimator, best_params).
    """
    import xgboost as xgb

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=search_iter,
        cv=3,
        scoring="f1_weighted",
        verbose=0,
        random_state=random_state,
        n_jobs=-1,
    )

    print("Training XGBoost with hyperparameter search...")
    search.fit(X_train, y_train)
    print(f"  Best F1 (CV): {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_iter: int = 20,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """Train Random Forest with randomised hyperparameter search.

    Returns (best_estimator, best_params).
    """
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    model = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=search_iter,
        cv=3,
        scoring="f1_weighted",
        verbose=0,
        random_state=random_state,
        n_jobs=-1,
    )

    print("Training Random Forest with hyperparameter search...")
    search.fit(X_train, y_train)
    print(f"  Best F1 (CV): {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------


def train_and_save(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    participant_array: np.ndarray,
    train_participants: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    search_iter: int = 20,
    random_state: int = 42,
    split_mode: str = "participant",
) -> Dict[str, Any]:
    """Run the full training pipeline and save models to disk.

    Parameters
    ----------
    split_mode : str
        ``"participant"`` — leave specified participants out for generalisation test.
        ``"random"`` — random stratified 80/20 split for in-distribution ceiling.
    """
    if output_dir is None:
        output_dir = MODEL_DIR

    if split_mode == "random":
        from sklearn.model_selection import train_test_split as tts
        X_all = feature_df.values.astype(np.float32)
        le = LabelEncoder()
        y_all = le.fit_transform(labels)
        X_train, X_test, y_train, y_test = tts(
            X_all, y_all, test_size=0.2, random_state=random_state, stratify=y_all
        )
        print(f"Random split — train: {len(X_train)}, test: {len(X_test)}")
        print(f"Classes: {list(le.classes_)}")
        train_parts = "random_80pct"
        test_parts = "random_20pct"
    else:
        X_train, X_test, y_train, y_test, le = prepare_data(
            feature_df, labels, participant_array, train_participants
        )
        train_parts = train_participants
        test_parts = [p for p in sorted(np.unique(participant_array)) if p not in (train_participants or [])]

    # --- XGBoost ---
    xgb_model, xgb_params = train_xgboost(X_train, y_train, search_iter, random_state)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)

    xgb_path = output_dir / "xgboost_model.pkl"
    with open(xgb_path, "wb") as f:
        pickle.dump({"model": xgb_model, "label_encoder": le, "params": xgb_params}, f)
    print(f"  ✓ Saved XGBoost to {xgb_path}")

    # --- Random Forest ---
    rf_model, rf_params = train_random_forest(X_train, y_train, search_iter // 2, random_state)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)

    rf_path = output_dir / "random_forest_model.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump({"model": rf_model, "label_encoder": le, "params": rf_params}, f)
    print(f"  ✓ Saved Random Forest to {rf_path}")

    # Save test data for downstream evaluation
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    print(f"  ✓ Saved test data to {output_dir}")

    # Save results JSON (without the large arrays)
    results_path = output_dir / "training_results.json"
    serialisable = {
        "split_mode": split_mode,
        "train_participants": str(train_parts),
        "test_participants": str(test_parts),
        "classes": list(le.classes_),
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "xgboost_params": xgb_params,
        "random_forest_params": rf_params,
    }
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  ✓ Saved training results to {results_path}")

    MODELS["xgboost"] = xgb_model
    MODELS["random_forest"] = rf_model
    MODELS["label_encoder"] = le
    MODELS["X_test"] = X_test
    MODELS["y_test"] = y_test

    return serialisable


def get_trained_models() -> Dict[str, Any]:
    """Return the last trained models (in-memory)."""
    return MODELS


if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.windowing import create_windows
    from src.features.features import extract_features, add_magnitude_features

    print("Loading data and extracting features...")
    df = load_data()
    windows, labels, meta = create_windows(df)
    windows_w_mag = add_magnitude_features(windows)
    feat_df, feat_names = extract_features(windows_w_mag, fs=100.0)

    print(f"Feature matrix: {feat_df.shape}")
    print(f"Labels: {labels.shape}")

    results = train_and_save(
        feature_df=feat_df,
        labels=meta["exercises"],
        participant_array=meta["participants"],
        # Leave P4 out as test (generalisation to unseen participant)
        train_participants=["P1", "P2", "P3"],
        search_iter=20,
    )
