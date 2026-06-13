"""Sliding-window segmentation for time-series sensor data."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Sensor channels used for modelling (excludes timestamp, participant, exercise)
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

# Default window parameters (at 100 Hz)
WINDOW_SIZE_S: float = 2.5
STRIDE_S: float = 1.25  # 50 % overlap


def create_windows(
    df: pd.DataFrame,
    window_size_s: float = WINDOW_SIZE_S,
    stride_s: float = STRIDE_S,
    fs: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Split the continuous sensor DataFrame into labelled sliding windows.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain SENSOR_CHANNELS, ``timestamp``, ``participant``, ``exercise``.
    window_size_s : float
        Window length in seconds.
    stride_s : float
        Step between consecutive windows in seconds.
    fs : float
        Sampling frequency (Hz) — used to convert seconds to samples.

    Returns
    -------
    windows : np.ndarray, shape (num_windows, window_size, num_channels)
        Raw sensor windows.
    labels : np.ndarray, shape (num_windows,)
        Integer-encoded exercise label for each window.
    metadata : dict
        ``participants``, ``exercises``, ``class_names``, ``window_starts``, ``window_ends``.
    """
    window_size = int(round(window_size_s * fs))
    stride = int(round(stride_s * fs))

    if window_size < 1 or stride < 1:
        raise ValueError("window_size_s and stride_s must produce at least 1 sample.")

    sensor_data = df[SENSOR_CHANNELS].values.astype(np.float64)
    n_total = len(sensor_data)

    windows_list: List[np.ndarray] = []
    labels_list: List[int] = []
    participants_list: List[str] = []
    exercises_list: List[str] = []
    starts: List[float] = []
    ends: List[float] = []

    # Group by participant + exercise to ensure windows don't cross boundaries
    grouped = df.groupby(["participant", "exercise"], sort=False)

    class_names = sorted(df["exercise"].unique())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for (participant, exercise), group_df in grouped:
        idx = class_to_idx[exercise]
        group_values = group_df[SENSOR_CHANNELS].values.astype(np.float64)
        group_len = len(group_values)
        timestamps = group_df["timestamp"].values

        n_windows_local = max(0, (group_len - window_size) // stride + 1)

        for w in range(n_windows_local):
            start = w * stride
            end = start + window_size
            windows_list.append(group_values[start:end])
            labels_list.append(idx)
            participants_list.append(participant)
            exercises_list.append(exercise)
            starts.append(float(timestamps[start]))
            ends.append(float(timestamps[end - 1]))

    windows = np.stack(windows_list, axis=0)
    labels = np.array(labels_list, dtype=np.int64)

    metadata: Dict[str, np.ndarray] = {
        "participants": np.array(participants_list, dtype=object),
        "exercises": np.array(exercises_list, dtype=object),
        "class_names": np.array(class_names, dtype=object),
        "window_starts": np.array(starts, dtype=np.float64),
        "window_ends": np.array(ends, dtype=np.float64),
    }

    return windows, labels, metadata


def create_windows_by_participant(
    df: pd.DataFrame,
    window_size_s: float = WINDOW_SIZE_S,
    stride_s: float = STRIDE_S,
    fs: float = 100.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
    """Create windows grouped per participant (useful for leave-one-subject-out)."""
    participants = df["participant"].unique()
    result: Dict[str, Tuple[np.ndarray, np.ndarray, Dict]] = {}
    for p in participants:
        p_df = df[df["participant"] == p].copy()
        w, l, m = create_windows(p_df, window_size_s, stride_s, fs)
        result[p] = (w, l, m)
    return result


if __name__ == "__main__":
    from src.data.load_data import load_data

    df = load_data()
    windows, labels, meta = create_windows(df)

    print(f"Windows shape: {windows.shape}")
    print(f"  -> {windows.shape[0]} windows, {windows.shape[1]} time-steps, {windows.shape[2]} channels")
    print(f"Labels shape: {labels.shape}")
    print(f"Classes: {meta['class_names']}")
    print(f"Label distribution:")
    for i, name in enumerate(meta["class_names"]):
        print(f"  {name}: {(labels == i).sum()} windows")
    print(f"\nParticipants present: {np.unique(meta['participants'])}")
