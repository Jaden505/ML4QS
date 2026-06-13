"""Sliding-window segmentation for time-series sensor data."""

import numpy as np
import pandas as pd

SENSOR_CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "lin_accel_x", "lin_accel_y", "lin_accel_z"]


def create_windows(df, window_size_s=2.5, stride_s=1.25, fs=100.0):
    """Split continuous sensor data into labelled sliding windows. Returns (windows, labels, metadata_dict)."""
    window_size = int(round(window_size_s * fs))
    stride = int(round(stride_s * fs))
    if window_size < 1 or stride < 1:
        raise ValueError("window_size_s and stride_s must produce at least 1 sample.")

    windows, labels, participants, exercises = [], [], [], []
    class_to_idx = {name: i for i, name in enumerate(sorted(df["exercise"].unique()))}

    for (participant, exercise), group in df.groupby(["participant", "exercise"], sort=False):
        vals = group[SENSOR_CHANNELS].values.astype(np.float64)
        n = len(vals)
        idx = class_to_idx[exercise]
        for w in range(max(0, (n - window_size) // stride + 1)):
            start = w * stride
            windows.append(vals[start:start + window_size])
            labels.append(idx)
            participants.append(participant)
            exercises.append(exercise)

    windows_arr = np.stack(windows, axis=0)
    labels_arr = np.array(labels, dtype=np.int64)
    meta = {"participants": np.array(participants, dtype=object), "exercises": np.array(exercises, dtype=object),
            "class_names": np.array(list(class_to_idx.keys()), dtype=object)}
    return windows_arr, labels_arr, meta


def create_windows_by_participant(df, window_size_s=2.5, stride_s=1.25, fs=100.0):
    """Create windows grouped per participant (useful for leave-one-subject-out)."""
    return {p: create_windows(df[df["participant"] == p], window_size_s, stride_s, fs) for p in df["participant"].unique()}


if __name__ == "__main__":
    from src.data.load_data import load_data
    df = load_data()
    windows, labels, meta = create_windows(df)
    print(f"Windows: {windows.shape} ({windows.shape[0]} windows, {windows.shape[1]} steps, {windows.shape[2]} channels)")
    for i, name in enumerate(meta["class_names"]):
        print(f"  {name}: {(labels == i).sum()} windows")
