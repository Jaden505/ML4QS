"""Time-domain and frequency-domain feature extraction from sensor windows."""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

SENSOR_CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "lin_accel_x", "lin_accel_y", "lin_accel_z"]


def _add_time_features(row, signal, name, fs):
    """Add mean, std, min, max, ptp, rms, mad, skew, kurtosis, zcr, sma, iqr, entropy."""
    s = signal.astype(np.float64)
    n = len(s)
    row[f"{name}_mean"] = float(np.mean(s))
    row[f"{name}_std"] = float(np.std(s, ddof=1))
    row[f"{name}_min"] = float(np.min(s))
    row[f"{name}_max"] = float(np.max(s))
    row[f"{name}_ptp"] = float(np.ptp(s))
    row[f"{name}_rms"] = float(np.sqrt(np.mean(s**2)))
    row[f"{name}_mad"] = float(np.mean(np.abs(s - np.mean(s))))
    row[f"{name}_skew"] = float(skew(s)) if n > 2 else 0.0
    row[f"{name}_kurtosis"] = float(kurtosis(s, fisher=True)) if n > 3 else 0.0
    row[f"{name}_zcr"] = float(np.sum(np.diff(np.signbit(s))) / n) if n > 0 else 0.0
    row[f"{name}_sma"] = float(np.sum(np.abs(s)) / n) if n > 0 else 0.0
    row[f"{name}_iqr"] = float(np.subtract(*np.percentile(s, [75, 25])))
    hist, _ = np.histogram(s, bins=20, density=True)
    hist = hist[hist > 0]
    row[f"{name}_entropy"] = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0


def _add_freq_features(row, signal, name, fs):
    """Add DC, peak freq, spectral energy, centroid, entropy, and 4 band ratios."""
    n = len(signal)
    if n < 4:
        for s in ["freq_dc", "freq_peak", "freq_energy", "freq_centroid", "freq_entropy", "freq_band1_ratio", "freq_band2_ratio", "freq_band3_ratio", "freq_band4_ratio"]:
            row[f"{name}_{s}"] = 0.0
        return
    fft_mag = np.abs(rfft(signal.astype(np.float64)))
    freqs = rfftfreq(n, d=1.0 / fs)
    valid = freqs > 0
    freqs_pos, mag_pos = freqs[valid], fft_mag[valid]
    row[f"{name}_freq_dc"] = float(fft_mag[0] / n)
    row[f"{name}_freq_peak"] = float(freqs_pos[np.argmax(mag_pos)]) if len(mag_pos) > 0 else 0.0
    row[f"{name}_freq_energy"] = float(np.sum(mag_pos**2) / n)
    row[f"{name}_freq_centroid"] = float(np.sum(freqs_pos * mag_pos) / np.sum(mag_pos)) if np.sum(mag_pos) > 0 else 0.0
    psd = mag_pos / np.sum(mag_pos) if np.sum(mag_pos) > 0 else mag_pos
    psd = psd[psd > 0]
    row[f"{name}_freq_entropy"] = float(-np.sum(psd * np.log2(psd))) if len(psd) > 0 else 0.0
    total_energy = np.sum(mag_pos**2) if len(mag_pos) > 0 else 1.0
    for i, (lo, hi) in enumerate([(0.5, 3.0), (3.0, 8.0), (8.0, 15.0), (15.0, fs / 2)]):
        in_band = (freqs_pos >= lo) & (freqs_pos < hi)
        band_e = np.sum(mag_pos[in_band]**2) if np.any(in_band) else 0.0
        row[f"{name}_freq_band{i+1}_ratio"] = float(band_e / total_energy) if total_energy > 0 else 0.0


def extract_features(windows, fs=100.0, channel_names=None):
    """Extract time + frequency features per window. Returns (feature_df, feature_names)."""
    n_windows, window_size, n_channels = windows.shape
    if channel_names is None:
        default = list(SENSOR_CHANNELS)
        while len(default) < n_channels:
            default.append(["accel_mag", "gyro_mag", "lin_accel_mag"][len(default) - len(SENSOR_CHANNELS)] if len(default) - len(SENSOR_CHANNELS) < 3 else f"extra_ch_{len(default) - len(SENSOR_CHANNELS)}")
        channel_names = default[:n_channels]
    rows = []
    for w in range(n_windows):
        row = {}
        for ch in range(n_channels):
            sig = windows[w, :, ch]
            _add_time_features(row, sig, channel_names[ch], fs)
            _add_freq_features(row, sig, channel_names[ch], fs)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, list(df.columns)


def add_magnitude_features(windows, channel_names=None):
    """Append accel_mag, gyro_mag, lin_accel_mag (Euclidean norm of each 3-axis group)."""
    n_windows, window_size, n_channels = windows.shape
    extra = []
    for ch_start in [0, 3, 6]:
        if ch_start + 2 < n_channels:
            mag = np.sqrt(windows[:, :, ch_start]**2 + windows[:, :, ch_start + 1]**2 + windows[:, :, ch_start + 2]**2)
            extra.append(mag[..., np.newaxis])
    return np.concatenate([windows] + extra, axis=2) if extra else windows


if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.windowing import create_windows
    df = load_data()
    windows, labels, meta = create_windows(df)
    windows_w_mag = add_magnitude_features(windows)
    feat_df, feat_names = extract_features(windows_w_mag, fs=100.0)
    print(f"Feature matrix: {feat_df.shape} ({len(feat_names)} features)")
