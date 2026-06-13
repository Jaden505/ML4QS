"""Time-domain and frequency-domain feature extraction from sensor windows."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SENSOR_CHANNELS: List[str] = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "lin_accel_x",
    "lin_accel_y",
    "lin_accel_z",
]


def extract_features(
    windows: np.ndarray,
    fs: float = 100.0,
    channel_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract time- and frequency-domain features from sensor windows.

    Parameters
    ----------
    windows : np.ndarray, shape (num_windows, window_size, num_channels)
    fs : float
        Sampling frequency (Hz).
    channel_names : list of str, optional
        Names for each channel (defaults to SENSOR_CHANNELS).

    Returns
    -------
    feature_df : pd.DataFrame, shape (num_windows, num_features)
    feature_names : list of str
        Ordered list of feature column names.
    """
    n_windows, window_size, n_channels = windows.shape
    if channel_names is None:
        default = list(SENSOR_CHANNELS)
        # Auto-generate names for any extra channels (e.g. magnitude features)
        while len(default) < n_channels:
            idx = len(default) - len(SENSOR_CHANNELS)
            prefix_list = ["accel_mag", "gyro_mag", "lin_accel_mag"]
            if idx < len(prefix_list):
                default.append(prefix_list[idx])
            else:
                default.append(f"extra_ch_{idx}")
        channel_names = default[:n_channels]

    feature_rows: List[Dict[str, float]] = []

    for w in range(n_windows):
        row: Dict[str, float] = {}
        for ch in range(n_channels):
            signal = windows[w, :, ch]
            name = channel_names[ch]
            _add_time_features(row, signal, name, fs)
            _add_freq_features(row, signal, name, fs)
        feature_rows.append(row)

    df = pd.DataFrame(feature_rows)
    return df, list(df.columns)


def extract_features_vectorized(
    windows: np.ndarray,
    fs: float = 100.0,
    channel_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Vectorised feature extraction — faster than per-window loops.

    Internally calls :func:`extract_features` (the loop implementation above
    is fast enough for ~thousands of windows, but this wrapper exists for
    consistency).
    """
    return extract_features(windows, fs, channel_names)


# ---------------------------------------------------------------------------
# Time-domain features
# ---------------------------------------------------------------------------

def _add_time_features(row: Dict[str, float], signal: np.ndarray, name: str, fs: float) -> None:
    """Add time-domain features for one channel to *row*."""
    s = signal.astype(np.float64)
    n = len(s)

    row[f"{name}_mean"] = float(np.mean(s))
    row[f"{name}_std"] = float(np.std(s, ddof=1))
    row[f"{name}_min"] = float(np.min(s))
    row[f"{name}_max"] = float(np.max(s))
    row[f"{name}_ptp"] = float(np.ptp(s))  # peak-to-peak amplitude
    row[f"{name}_rms"] = float(np.sqrt(np.mean(s**2)))
    row[f"{name}_mad"] = float(np.mean(np.abs(s - np.mean(s))))  # mean absolute deviation
    row[f"{name}_skew"] = float(skew(s)) if n > 2 else 0.0
    row[f"{name}_kurtosis"] = float(kurtosis(s, fisher=True)) if n > 3 else 0.0

    # Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(s)))
    row[f"{name}_zcr"] = float(zero_crossings) / n if n > 0 else 0.0

    # Signal magnitude area (SMA) — already per-channel here
    row[f"{name}_sma"] = float(np.sum(np.abs(s))) / n if n > 0 else 0.0

    # Interquartile range
    row[f"{name}_iqr"] = float(np.subtract(*np.percentile(s, [75, 25])))

    # Entropy proxy — histogram-based
    hist, _ = np.histogram(s, bins=20, density=True)
    hist = hist[hist > 0]
    row[f"{name}_entropy"] = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0


# ---------------------------------------------------------------------------
# Frequency-domain features
# ---------------------------------------------------------------------------

def _add_freq_features(row: Dict[str, float], signal: np.ndarray, name: str, fs: float) -> None:
    """Add frequency-domain features for one channel."""
    n = len(signal)
    if n < 4:
        # Not enough samples for meaningful FFT
        for suffix in ["freq_dc", "freq_peak", "freq_energy", "freq_centroid", "freq_entropy"]:
            row[f"{name}_{suffix}"] = 0.0
        return

    fft_vals = rfft(signal.astype(np.float64))
    fft_mag = np.abs(fft_vals)
    freqs = rfftfreq(n, d=1.0 / fs)

    idx_valid = freqs > 0  # exclude DC for many calculations
    freqs_pos = freqs[idx_valid]
    mag_pos = fft_mag[idx_valid]

    # DC component
    row[f"{name}_freq_dc"] = float(fft_mag[0]) / n if n > 0 else 0.0

    # Dominant (peak) frequency
    if len(mag_pos) > 0:
        peak_idx = np.argmax(mag_pos)
        row[f"{name}_freq_peak"] = float(freqs_pos[peak_idx])
    else:
        row[f"{name}_freq_peak"] = 0.0

    # Total spectral energy
    row[f"{name}_freq_energy"] = float(np.sum(mag_pos**2)) / n if n > 0 else 0.0

    # Spectral centroid (weighted mean frequency)
    if np.sum(mag_pos) > 0:
        row[f"{name}_freq_centroid"] = float(np.sum(freqs_pos * mag_pos) / np.sum(mag_pos))
    else:
        row[f"{name}_freq_centroid"] = 0.0

    # Spectral entropy
    if np.sum(mag_pos) > 0:
        psd = mag_pos / np.sum(mag_pos)
        psd = psd[psd > 0]
        row[f"{name}_freq_entropy"] = float(-np.sum(psd * np.log2(psd))) if len(psd) > 0 else 0.0
    else:
        row[f"{name}_freq_entropy"] = 0.0

    # Energy in frequency bands (useful for distinguishing exercises)
    # Band 1: 0.5–3 Hz (slow movements)
    # Band 2: 3–8 Hz  (moderate)
    # Band 3: 8–15 Hz (fast/jerky)
    # Band 4: 15+ Hz  (noise / sharp transients)
    bands = [(0.5, 3.0), (3.0, 8.0), (8.0, 15.0), (15.0, fs / 2)]
    total_energy = np.sum(mag_pos**2) if len(mag_pos) > 0 else 1.0
    for i, (lo, hi) in enumerate(bands):
        in_band = (freqs_pos >= lo) & (freqs_pos < hi)
        band_energy = np.sum(mag_pos[in_band] ** 2) if np.any(in_band) else 0.0
        row[f"{name}_freq_band{i+1}_ratio"] = float(band_energy / total_energy) if total_energy > 0 else 0.0


# ---------------------------------------------------------------------------
# Convenience: combined magnitude features (across axes)
# ---------------------------------------------------------------------------

def add_magnitude_features(
    windows: np.ndarray, channel_names: Optional[List[str]] = None
) -> np.ndarray:
    """Compute signal magnitude (Euclidean norm) for 3-axis sensor groups.

    Returns extra channels: accel_mag, gyro_mag, lin_accel_mag appended to windows.
    """
    if channel_names is None:
        channel_names = SENSOR_CHANNELS

    n_windows, window_size, n_channels = windows.shape
    extra: List[np.ndarray] = []

    groups = [("accel", 0), ("gyro", 3), ("lin_accel", 6)]
    for prefix, ch_start in groups:
        if ch_start + 2 < n_channels:
            x = windows[:, :, ch_start]
            y = windows[:, :, ch_start + 1]
            z = windows[:, :, ch_start + 2]
            mag = np.sqrt(x**2 + y**2 + z**2)
            extra.append(mag[..., np.newaxis])  # (N, T) → (N, T, 1)

    return np.concatenate([windows] + extra, axis=2) if extra else windows


if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.windowing import create_windows

    df = load_data()
    windows, labels, meta = create_windows(df)

    # Add magnitude channels and extract features
    windows_w_mag = add_magnitude_features(windows)
    feat_df, feat_names = extract_features(windows_w_mag, fs=100.0)

    print(f"Feature matrix shape: {feat_df.shape}")
    print(f"Number of features: {len(feat_names)}")
    print(f"\nFirst 5 feature names:")
    for n in feat_names[:10]:
        print(f"  {n}")
    print(f"  ... ({len(feat_names)} total)")
    print(f"\nFeature matrix preview:")
    print(feat_df.head())
    print(f"\nFeature statistics:")
    print(feat_df.describe())
