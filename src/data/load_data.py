"""Load, flatten, and resample merged inertial sensor HDF5 data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

# Sensor configuration: (group_name, channel_prefix, channel_count)
SENSOR_CONFIG: List[Tuple[str, str, int]] = [
    ("accelerometer", "accel", 3),
    ("gyroscope", "gyro", 3),
    ("linear_accelerometer", "lin_accel", 3),
]

AXIS_SUFFIXES = ["_x", "_y", "_z"]

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "merged_interial_sensor_data.h5"


def _discover_sessions(h5_file: h5py.File) -> List[Tuple[str, str]]:
    """Discover all (participant, exercise) pairs in the HDF5 file."""
    sessions: List[Tuple[str, str]] = []
    for participant_key in h5_file.keys():
        participant_group = h5_file[participant_key]
        for exercise_key in participant_group.keys():
            # Skip non-exercise groups (device_metadata, etc.)
            exercise_group = participant_group[exercise_key]
            if "accelerometer" in exercise_group:
                sessions.append((participant_key, exercise_key))
    return sorted(sessions)


def _read_sensor_data(
    sensor_group, uniform_time: np.ndarray
) -> Dict[str, np.ndarray]:
    """Resample all axes of a sensor to a uniform time axis."""
    columns_attr = sensor_group.attrs.get("columns", [])
    if isinstance(columns_attr, bytes):
        columns_attr = [columns_attr]

    # Extract time and axis data
    axis_names = [c for c in columns_attr if c != "time_s"]
    time_data = sensor_group["time_s"][:]

    result: Dict[str, np.ndarray] = {}
    for axis in axis_names:
        raw = sensor_group[axis][:]
        # Only interpolate if we have enough points
        if len(time_data) > 1:
            interpolator = interp1d(
                time_data, raw, kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            result[str(axis)] = interpolator(uniform_time)
        else:
            result[str(axis)] = np.full_like(uniform_time, raw[0] if len(raw) > 0 else 0.0)

    return result


def _load_session(
    h5_file: h5py.File, participant: str, exercise: str, target_fs: float = 100.0
) -> Optional[pd.DataFrame]:
    """Load and resample a single (participant, exercise) session."""
    session_path = f"{participant}/{exercise}"

    # Determine session duration from accelerometer (reference sensor)
    accel_group = h5_file[f"{session_path}/accelerometer"]
    accel_time = accel_group["time_s"][:]
    session_duration = float(accel_time[-1])
    n_samples = int(np.floor(session_duration * target_fs))

    if n_samples < 10:
        return None  # Skip degenerate sessions

    uniform_time = np.linspace(0, session_duration, n_samples, endpoint=False)

    # Build the DataFrame row by row for this session
    data: Dict[str, np.ndarray] = {}
    for sensor_name, prefix, _ in SENSOR_CONFIG:
        sensor_path = f"{session_path}/{sensor_name}"
        if sensor_path not in h5_file:
            continue
        sensor_data = _read_sensor_data(h5_file[sensor_path], uniform_time)
        for axis_key, values in sensor_data.items():
            col_name = f"{prefix}_{axis_key.split('_')[-1]}"
            data[col_name] = values

    df = pd.DataFrame(data)
    df["timestamp"] = uniform_time
    df["participant"] = participant
    df["exercise"] = exercise
    return df


def load_data(
    h5_path: str = "data/merged_interial_sensor_data.h5",
    target_fs: float = 100.0,
) -> pd.DataFrame:
    """Load and flatten the merged inertial sensor HDF5 file into a DataFrame.

    Each sensor channel is resampled to *target_fs* Hz using linear interpolation.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, participant, exercise,
        accel_x, accel_y, accel_z,
        gyro_x, gyro_y, gyro_z,
        lin_accel_x, lin_accel_y, lin_accel_z
    """
    h5_path_resolved = str(Path(h5_path).resolve())
    if not Path(h5_path_resolved).exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path_resolved}")

    with h5py.File(h5_path_resolved, "r") as f:
        sessions = _discover_sessions(f)

        frames: List[pd.DataFrame] = []
        for participant, exercise in tqdm(sessions, desc="Loading sessions"):
            df = _load_session(f, participant, exercise, target_fs)
            if df is not None:
                frames.append(df)

    if not frames:
        raise ValueError("No valid sessions found in the HDF5 file.")

    result = pd.concat(frames, ignore_index=True)

    # Sort for consistent ordering
    result.sort_values(["participant", "exercise", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    return result


if __name__ == "__main__":
    df = load_data()
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Participants: {df['participant'].unique()}")
    print(f"Exercises: {df['exercise'].unique()}")
    print(f"Sampling rate check:\n{df.groupby(['participant', 'exercise'])['timestamp'].agg(['count', 'min', 'max'])}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
