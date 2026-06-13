<!-- Context: project-intelligence/technical | Priority: critical | Version: 1.1 | Updated: 2026-06-13 -->

# Technical Domain

**Purpose**: Tech stack, architecture, and development patterns for the ML4QS project.
**Last Updated**: 2026-06-13

## Quick Reference
- **Update Triggers**: New libraries, pattern changes, architecture decisions
- **Audience**: Developers, AI agents building ML/data-analysis features
- **Domain**: Machine Learning for the Quantified Self (sensor data analysis)

## Primary Stack

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| Language | Python | ≥3.11 | ML ecosystem maturity |
| Data Processing | Pandas | ≥3.0.3 | DataFrame-centric analysis |
| Numerical | NumPy | ≥2.4.6 | Array operations |
| ML/Classification | Scikit-learn | latest | Standard ML algorithms |
| Deep Learning | PyTorch (w/ Transformer) | latest | State-of-the-art sequence modeling |
| Visualization | Matplotlib | ≥3.11.0 | Publication-quality plots |
| Environment | venv (`.venv`) | system | Dependency isolation |

## Project: Exercise Classification

### Data Collection
- **App**: phyphox (smartphone sensor recording)
- **Phone placement**: Strapped to forearm (see project report Fig. 1)
- **Sampling**: phyphox at 0 Hz setting (max rate) → effective ~100–500 Hz
- **Duration**: 5 minutes per exercise per participant
- **Participants**: 4 (Ahmad Ramadan, Taro Schenk, Jaden van Rijswijk, +1)

### Sensor Modalities

| Sensor | Axes | Measures | Gravity? |
|--------|------|----------|----------|
| Accelerometer | x, y, z | Acceleration including gravity | ✅ Included |
| Gyroscope | x, y, z | Angular velocity | N/A |
| Linear Acceleration | x, y, z | Acceleration minus gravity | ❌ Removed |

### Exercises (Target Classes)

| Exercise | Movement Pattern | Arm Motion | Body Motion |
|----------|-----------------|------------|-------------|
| Bicep Curls | Half-circle rotation | Forearm only | Stationary |
| Shoulder Presses | Vertical push above shoulder | Full arm up/down | Minimal |
| Jumping Jacks | Fast large arm circles + jump | Full arm | ✅ Jumping |

### Model Approaches

| Approach | Models | Feature Type |
|----------|--------|-------------|
| **Classical ML** (cf. Chapter 7) | SVM, Random Forest, etc. | Handcrafted features (cf. Chapter 4) |
| **Deep Learning** (state-of-the-art) | Transformer (self-attention), LSTM, or TCN | Learned from windowed raw sensor data |

### Evaluation Setup
- **Windowing**: 2–3 second overlapping windows
- **Train/Test Split**: Generalization-focused (across persons or new recordings)
- **NOT**: Random timepoint sampling from single recording
- **Ablation**: Test each sensor modality individually for RQ2

## Architecture Pattern

```
Type: Modular Script-based + DL Pipeline
Pattern: Chapter-organized scripts + custom DL model training
Data Flow: phyphox CSV → Resampling → Windowing → Feature Engineering → Classical ML
                                                          ↓
                                               Raw Windows → Transformer/LSTM → Classification
```

## Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Files/Dirs | `snake_case` | `dataloader.py`, `crowdsignals_ch2.py` |
| Classes | `PascalCase` | `CreateDataset`, `VisualizeDataset` |
| Functions | `snake_case` | `normalize_dataset()`, `print_statistics()` |
| Constants | `UPPER_CASE` | `DATASET_PATH`, `GRANULARITIES` |
| Variables | `snake_case` | `data_table`, `milliseconds_per_instance` |

## Code Standards

- **Type hints** encouraged (Python 3.11+)
- **Modular structure** — logic split by chapter/concern into separate files
- **Utility modules** — shared functions in `util/` directory
- **Path constants** — defined at top of scripts (`DATASET_PATH`, `RESULT_PATH`)
- **Comments explain "why"** not "what" (code is self-documenting for mechanics)
- **Data flows through Pandas DataFrames** as the core data structure
- **Virtual environment** (`.venv`) for dependency isolation
- **Docker support** via `Dockerfile` for reproducible execution
- **Dependencies pinned** in `requirements.txt` and `pyproject.toml`
- **Granularity-driven** — time series processing with configurable `milliseconds_per_instance`

## Security Requirements

- **Data files tracked in git** — datasets are committed alongside code
- **Virtual environment isolation** — dependencies managed via `.venv`
- **File path sanitization** — use `Path(__file__).parent` for relative paths
- **Dependency pinning** — versions locked in `requirements.txt` and `pyproject.toml`

## 📂 Codebase References

| File | Purpose |
|------|---------|
| `Chapters/crowdsignals_ch2.py` | Ch.2 data loading & exploration reference |
| `Chapters/crowdsignals_ch3_outliers.py` | Ch.3 noise & outlier handling reference |
| `Chapters/crowdsignals_ch4.py` | Ch.4 feature engineering reference |
| `Chapters/crowdsignals_ch7_classification.py` | Ch.7 classical ML classification reference |
| `Chapters/crowdsignals_ch8_regression.py` | Ch.8 deep learning / regression reference |
| `Chapters/Chapter2/CreateDataset.py` | Data loading class (sensor CSV → DataFrame) |
| `Chapters/util/util.py` | Shared utilities (normalize, distance, stats) |
| `Chapters/util/VisualizeDataset.py` | Visualization utility class |
| `pyproject.toml` | Project metadata + dependencies |
| `Dockerfile` | Containerized execution setup |

## Related Files
- `business-domain.md` — Assignment context, RQs, deadlines
- `business-tech-bridge.md` — How business needs map to technical solutions
- `decisions-log.md` — Key decisions and rationale
- `living-notes.md` — Active issues and open questions
