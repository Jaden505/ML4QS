"""Load, flatten, and resample merged inertial sensor HDF5 data."""

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

SENSOR_CONFIG = [("accelerometer", "accel", 3), ("gyroscope", "gyro", 3), ("linear_accelerometer", "lin_accel", 3)]
DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "merged_interial_sensor_data.h5"
SENSOR_CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
                   "lin_accel_x", "lin_accel_y", "lin_accel_z"]

def _discover_sessions(h5_file):
    """Find all (participant, exercise) pairs in the HDF5 file."""
    sessions = []
    for p_key in h5_file.keys():
        for e_key in h5_file[p_key].keys():
            if "accelerometer" in h5_file[p_key][e_key]:
                sessions.append((p_key, e_key))
    return sorted(sessions)


def _read_sensor_data(sensor_group, uniform_time):
    """Resample all axes of a sensor to a uniform time axis via linear interpolation."""
    cols = sensor_group.attrs.get("columns", [])
    if isinstance(cols, bytes): cols = [cols]
    axis_names = [c for c in cols if c != "time_s"]
    time_data = sensor_group["time_s"][:]
    result = {}
    for axis in axis_names:
        raw = sensor_group[axis][:]
        if len(time_data) > 1:
            interp = interp1d(time_data, raw, kind="linear", bounds_error=False, fill_value="extrapolate")
            result[str(axis)] = interp(uniform_time)
        else:
            result[str(axis)] = np.full_like(uniform_time, raw[0] if len(raw) > 0 else 0.0)
    return result



def _load_session(h5_file, participant, exercise, target_fs=100.0):
    """Load and resample a single (participant, exercise) session."""
    accel_time = h5_file[f"{participant}/{exercise}/accelerometer"]["time_s"][:]
    n_samples = int(np.floor(float(accel_time[-1]) * target_fs))
    if n_samples < 10: return None
    uniform_time = np.linspace(0, float(accel_time[-1]), n_samples, endpoint=False)
    data = {}
    for sensor_name, prefix, _ in SENSOR_CONFIG:
        path = f"{participant}/{exercise}/{sensor_name}"
        if path not in h5_file: continue
        sensor_data = _read_sensor_data(h5_file[path], uniform_time)
        for k, v in sensor_data.items():
            data[f"{prefix}_{k.split('_')[-1]}"] = v
    df = pd.DataFrame(data)
    df["timestamp"] = uniform_time
    df["participant"] = participant
    df["exercise"] = exercise
    return df


def _normalize_per_participant(df, method="zscore"):
    """Z-score normalize each sensor channel per participant to reduce device bias."""
    result = df.copy()
    for participant in df["participant"].unique():
        for ch in SENSOR_CHANNELS:
            vals = df.loc[df["participant"] == participant, ch]
            result.loc[df["participant"] == participant, ch] = (vals - vals.mean()) / (vals.std() + 1e-8)
    return result


def load_data(h5_path=str(DATASET_PATH), target_fs=100.0):
    """Flatten HDF5 into a DataFrame with all sensor channels resampled to target_fs Hz."""
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        frames = []
        for participant, exercise in tqdm(_discover_sessions(f), desc="Loading sessions"):
            df = _load_session(f, participant, exercise, target_fs)
            if df is not None: frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["participant", "exercise", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    result = _normalize_per_participant(result, method="zscore")
    return result


if __name__ == "__main__":
    df = load_data()
    print(f"Shape: {df.shape}\nColumns: {df.columns.tolist()}\nParticipants: {df['participant'].unique()}\nExercises: {df['exercise'].unique()}")
