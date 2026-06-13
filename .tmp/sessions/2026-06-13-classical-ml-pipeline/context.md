# Task Context: Classical ML Pipeline for Exercise Classification

Session ID: 2026-06-13-classical-ml-pipeline
Created: 2026-06-13
Status: in_progress

## Current Request
Build a complete classical ML pipeline for the merged_inertial_sensor HDF5 data:
1. Load and flatten the HDF5 data into a usable DataFrame
2. Resample to consistent frequency and window into segments
3. Engineer handcrafted time/frequency features
4. Train classical ML models (XGBoost + Random Forest) for exercise classification
5. Experiment with data augmentation
6. Evaluate with concise metrics (precision/recall/F1, confusion matrix)
7. Extract feature importance insights (SHAP + RF importance)
8. Create a blueprint script for future Transformer training

## Context Files (Standards to Follow)
- .opencode/context/core/standards/code-quality.md
- .opencode/context/project-intelligence/technical-domain.md
- .opencode/context/project-intelligence/business-domain.md

## Reference Files (Source Material to Look At)
- data/merged_interial_sensor_data.h5
- pyproject.toml

## External Docs Fetched
- scikit-learn (available, installed via pip)
- xgboost (to be installed)
- shap (to be installed)

## Components

| # | Component | Description |
|---|-----------|-------------|
| 1 | `load_data.py` | Flatten HDF5 → DataFrame, resample to 100Hz, add participant/exercise labels |
| 2 | `windowing.py` | Sliding window segmentation (2.5s, 50% overlap), produce (windows × channels) |
| 3 | `features.py` | Time-domain + frequency-domain feature extraction per window |
| 4 | `train_classical.py` | Train XGBoost + Random Forest, hyperparameter tuning, evaluate |
| 5 | `augmentation.py` | Data augmentation (noise, time-warp, scaling, sensor dropout) + retrain |
| 6 | `evaluate.py` | Metrics (precision/recall/F1), confusion matrix, per-class breakdown |
| 7 | `feature_insights.py` | SHAP values, RF feature importance, top feature analysis with viz |
| 8 | `transformer_blueprint.py` | PyTorch Transformer training scaffold (raw window → Transformer → classification) |
| 9 | `run_pipeline.py` | Orchestrator that runs the full pipeline end-to-end |

## Data Structure (from HDF5 exploration)

```
Hierarchy: Participant (P1-P4) → Exercise (bicep_curl, jumping_jacks, shoulder_press) → Sensor (accelerometer, gyroscope, linear_accelerometer) → axis signals

Sensors:
  - accelerometer:  accel_x, accel_y, accel_z  (m/s², includes gravity)
  - gyroscope:      gyro_x, gyro_y, gyro_z     (rad/s)
  - linear_accelerometer: lin_accel_x, lin_accel_y, lin_accel_z (m/s², gravity removed)

Sampling rates:
  - P1 (iPhone13,2): ~100 Hz
  - P2 (iPhone14,8): ~100 Hz
  - P3 (iPhone13,4): ~100 Hz
  - P4 (SM-G991B):   ~500 Hz (accel/gyro), ~100 Hz (lin_accel)

Total: 4 participants × 3 exercises = 12 sessions
```

## Constraints
- Classical ML only (XGBoost + Random Forest)
- Train/test split by PARTICIPANT (not random timepoints) — generalization test
- Window size: 2.5 seconds (250 samples at 100Hz), 50% overlap
- Output scripts in project root (classical_ml/ directory)
- Visualization saves to figures/ directory
- Pip install xgboost, shap if not already installed

## Exit Criteria
- [ ] All 9 components script files created
- [ ] Pipeline runs end-to-end without errors
- [ ] Classification metrics (F1 ≥ 0.85 target) produced
- [ ] Feature importance analysis with SHAP plots saved
- [ ] Augmentation comparison table produced
- [ ] Transformer blueprint scaffold exists
