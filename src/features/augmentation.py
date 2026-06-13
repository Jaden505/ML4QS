"""Time-series data augmentation for sensor windows."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def add_gaussian_noise(window, noise_level=0.05):
    """Add Gaussian noise scaled by each channel's standard deviation."""
    return window + np.random.normal(0, noise_level, size=window.shape) * (np.std(window, axis=0, keepdims=True) + 1e-8)


def time_warp(window, stretch=1.0):
    """Apply smooth non-linear time distortion via random displacement."""
    t = np.arange(window.shape[0], dtype=np.float64)
    d = np.random.uniform(-0.5, 0.5, size=window.shape[0])
    
    kernel = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
    d = np.convolve(d, kernel / kernel.sum(), mode="same")
    d = (d - d[0]) * stretch * 5.0
    
    t_w = np.clip(t + d, 0, window.shape[0] - 1)
    result = np.zeros_like(window)
    for ch in range(window.shape[1]):
        result[:, ch] = np.interp(t_w, t, window[:, ch])
        
    return result


def amplitude_scale(window, scale_range=(0.7, 1.3)):
    """Scale amplitude by random per-channel factors."""
    return window * np.random.uniform(*scale_range, size=(1, window.shape[1]))


def sensor_dropout(window, dropout_prob=0.15, fill_value=0.0):
    """Randomly zero out entire channels."""
    result = window.copy()
    result[:, np.random.rand(window.shape[1]) > dropout_prob] = fill_value
    return result


def time_shift(window, fs=100.0):
    """Circular shift by up to 15% of window length."""
    max_shift = int(window.shape[0] * 0.15)
    shift = np.random.randint(-max_shift, max_shift + 1) if max_shift > 0 else 0
    return np.roll(window, shift, axis=0) if shift != 0 else window.copy()


AUGMENTATIONS = {"noise": add_gaussian_noise, "warp": time_warp, "scale": amplitude_scale, "dropout": sensor_dropout, "shift": time_shift}


def augment_window(window, techniques=None, apply_prob=0.5):
    """Apply random subset of augmentation techniques to a single window."""
    if techniques is None: techniques = list(AUGMENTATIONS.keys())
    aug = window.copy()
    for name in techniques:
        if np.random.rand() < apply_prob:
            aug = AUGMENTATIONS[name](aug)
    return aug


def augment_dataset(windows, labels, augment_factor=2, techniques=None, apply_prob=0.5, seed=None):
    """Generate augmented copies, returns (original + augmented) stacked."""
    if seed is not None: np.random.seed(seed)
    n = len(windows)
    aug_w = [windows]
    aug_l = [labels]
    for _ in range(augment_factor):
        batch = np.array([augment_window(windows[i], techniques, apply_prob) for i in range(n)])
        aug_w.append(batch)
        aug_l.append(labels.copy())
    return np.concatenate(aug_w, axis=0), np.concatenate(aug_l, axis=0)


def compare_augmentation(windows, labels, train_idx, test_idx, augment_factor=2, techniques=None, apply_prob=0.5):
    """Quick RF comparison of augmented vs non-augmented data."""
    ws = windows.shape[1] * windows.shape[2]
    X_train, X_test = windows.reshape(len(windows), ws)[train_idx], windows.reshape(len(windows), ws)[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    base = {"accuracy": accuracy_score(y_test, pred), "f1_weighted": f1_score(y_test, pred, average="weighted")}
    
    aug_w, aug_l = augment_dataset(windows[train_idx], labels[train_idx], augment_factor, techniques, apply_prob)
    rf2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf2.fit(aug_w.reshape(len(aug_w), ws), aug_l)
    pred2 = rf2.predict(X_test)
    
    aug = {"accuracy": accuracy_score(y_test, pred2), "f1_weighted": f1_score(y_test, pred2, average="weighted")}
    return {**{"baseline_" + k: v for k, v in base.items()}, **{"augmented_" + k: v for k, v in aug.items()}, "augment_factor": augment_factor}


if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.windowing import create_windows
    from sklearn.model_selection import train_test_split
    df = load_data()
    windows, labels, meta = create_windows(df)
    aug_w, aug_l = augment_dataset(windows, labels, augment_factor=2)
    print(f"Original: {windows.shape[0]} windows, Augmented: {aug_w.shape[0]} windows")
    train_idx, test_idx = train_test_split(np.arange(len(windows)), test_size=0.2, random_state=42, stratify=labels)
    results = compare_augmentation(windows, labels, train_idx, test_idx)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
