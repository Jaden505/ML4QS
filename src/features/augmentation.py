"""Time-series data augmentation for sensor windows."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

AugFunc = Callable[[np.ndarray], np.ndarray]

# ---------------------------------------------------------------------------
# Augmentation primitives  (each operates on a single window)
# ---------------------------------------------------------------------------


def add_gaussian_noise(
    window: np.ndarray, noise_level: float = 0.05
) -> np.ndarray:
    """Add isotropic Gaussian noise scaled by the window's standard deviation."""
    std = np.std(window, axis=0, keepdims=True)
    noise = np.random.normal(0, noise_level, size=window.shape) * (std + 1e-8)
    return window + noise


def time_warp(
    window: np.ndarray, stretch_factor_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """Apply a random smooth time-warp via cubic interpolation.

    Each time-step is perturbed by a small random offset, creating a
    non-linear temporal distortion.
    """
    t_orig = np.arange(window.shape[0], dtype=np.float64)
    warp_strength = np.random.uniform(*stretch_factor_range)

    # Generate random displacement field and smooth it
    displacement = np.random.uniform(-0.5, 0.5, size=window.shape[0])
    # Simple triangular smoothing
    kernel = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
    kernel = kernel / kernel.sum()
    displacement = np.convolve(displacement, kernel, mode="same")
    displacement = displacement - displacement[0]  # fix start point
    displacement *= warp_strength * 5.0

    t_warped = t_orig + displacement
    t_warped = np.clip(t_warped, 0, window.shape[0] - 1)

    # Linear interpolation at warped positions
    result = np.zeros_like(window)
    for ch in range(window.shape[1]):
        result[:, ch] = np.interp(t_warped, t_orig, window[:, ch])
    return result


def amplitude_scale(
    window: np.ndarray, scale_range: Tuple[float, float] = (0.7, 1.3)
) -> np.ndarray:
    """Scale the amplitude by a random factor (per-channel)."""
    scales = np.random.uniform(*scale_range, size=(1, window.shape[1]))
    return window * scales


def sensor_dropout(
    window: np.ndarray, dropout_prob: float = 0.15, fill_value: float = 0.0
) -> np.ndarray:
    """Randomly set a fraction of channels to *fill_value* for the whole window."""
    mask = np.random.rand(window.shape[1]) > dropout_prob
    result = window.copy()
    result[:, ~mask] = fill_value
    return result


def time_shift(
    window: np.ndarray, shift_range_s: Tuple[float, float] = (-0.3, 0.3), fs: float = 100.0
) -> np.ndarray:
    """Circularly shift the window forward/backward in time."""
    max_shift = int(window.shape[0] * 0.15)  # at most 15 % of window
    shift = np.random.randint(-max_shift, max_shift + 1) if max_shift > 0 else 0
    if shift == 0:
        return window.copy()
    return np.roll(window, shift, axis=0)


# ---------------------------------------------------------------------------
# Composable augmentation pipeline
# ---------------------------------------------------------------------------

AUGMENTATION_REGISTRY: Dict[str, AugFunc] = {
    "noise": lambda w, nl=0.05: add_gaussian_noise(w, nl),
    "warp": lambda w: time_warp(w),
    "scale": lambda w: amplitude_scale(w),
    "dropout": lambda w: sensor_dropout(w),
    "shift": lambda w: time_shift(w),
}


def augment_window(
    window: np.ndarray,
    techniques: Optional[List[str]] = None,
    apply_prob: float = 0.5,
) -> np.ndarray:
    """Apply a random subset of augmentation techniques to a single window.

    Parameters
    ----------
    window : np.ndarray, shape (window_size, num_channels)
    techniques : list of str, optional
        Which techniques to consider (default: all).
    apply_prob : float
        Probability that a given technique is applied to the window.

    Returns
    -------
    np.ndarray — augmented window (same shape).
    """
    if techniques is None:
        techniques = list(AUGMENTATION_REGISTRY.keys())

    aug = window.copy()
    for name in techniques:
        if np.random.rand() < apply_prob:
            fn = AUGMENTATION_REGISTRY[name]
            aug = fn(aug)
    return aug


def augment_dataset(
    windows: np.ndarray,
    labels: np.ndarray,
    augment_factor: int = 2,
    techniques: Optional[List[str]] = None,
    apply_prob: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment the entire windowed dataset by generating synthetic copies.

    Parameters
    ----------
    windows : np.ndarray, shape (N, window_size, num_channels)
    labels : np.ndarray, shape (N,)
    augment_factor : int
        How many augmented copies to generate (e.g., 2 → 2× original size).
    techniques, apply_prob : passed to ``augment_window``.

    Returns
    -------
    combined_windows, combined_labels — original + augmented data stacked.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(windows)
    augmented_windows: List[np.ndarray] = []
    augmented_labels: List[np.ndarray] = []

    for _ in range(augment_factor):
        batch = np.array([augment_window(windows[i], techniques, apply_prob) for i in range(n)])
        augmented_windows.append(batch)
        augmented_labels.append(labels.copy())

    all_windows = np.concatenate([windows] + augmented_windows, axis=0)
    all_labels = np.concatenate([labels] + augmented_labels, axis=0)

    return all_windows, all_labels


def compare_augmentation(
    windows: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    augment_factor: int = 2,
    techniques: Optional[List[str]] = None,
    apply_prob: float = 0.5,
) -> Dict[str, float]:
    """Train a quick RF on augmented vs non-augmented data and return scores.

    Requires scikit-learn.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    # Flatten windows for sklearn
    n_w, ws, nc = windows.shape
    X_flat = windows.reshape(n_w, ws * nc)

    X_train, y_train = X_flat[train_idx], labels[train_idx]
    X_test, y_test = X_flat[test_idx], labels[test_idx]

    # --- Baseline (no augmentation) ---
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    pred_base = rf_base.predict(X_test)
    base_acc = accuracy_score(y_test, pred_base)
    base_f1 = f1_score(y_test, pred_base, average="weighted")

    # --- With augmentation ---
    aug_w, aug_l = augment_dataset(
        windows[train_idx],
        labels[train_idx],
        augment_factor=augment_factor,
        techniques=techniques,
        apply_prob=apply_prob,
    )
    X_aug = aug_w.reshape(len(aug_w), ws * nc)

    rf_aug = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_aug.fit(X_aug, aug_l)
    pred_aug = rf_aug.predict(X_test)
    aug_acc = accuracy_score(y_test, pred_aug)
    aug_f1 = f1_score(y_test, pred_aug, average="weighted")

    return {
        "baseline_accuracy": base_acc,
        "baseline_f1_weighted": base_f1,
        "augmented_accuracy": aug_acc,
        "augmented_f1_weighted": aug_f1,
        "augment_factor": augment_factor,
    }


if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.windowing import create_windows

    df = load_data()
    windows, labels, meta = create_windows(df)

    # Demonstrate augmentation on a single window
    single = windows[0]
    aug_single = augment_window(single, apply_prob=1.0)
    print(f"Original window shape: {single.shape}")
    print(f"Augmented window shape: {aug_single.shape}")
    print(f"Original mean: {single.mean():.4f}, Augmented mean: {aug_single.mean():.4f}")

    # Augment full dataset
    aug_windows, aug_labels = augment_dataset(windows, labels, augment_factor=2)
    print(f"\nOriginal: {windows.shape[0]} windows, Augmented: {aug_windows.shape[0]} windows")

    # Quick comparison on 80/20 split
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        np.arange(len(windows)), test_size=0.2, random_state=42, stratify=labels
    )
    results = compare_augmentation(windows, labels, train_idx, test_idx)
    print(f"\nAugmentation comparison:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
