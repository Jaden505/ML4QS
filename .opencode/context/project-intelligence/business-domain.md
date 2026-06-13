<!-- Context: project-intelligence/business | Priority: critical | Version: 1.0 | Updated: 2026-06-13 -->

# Business Domain

**Purpose**: Assignment context, research questions, and project scope for the ML4QS course project.
**Last Updated**: 2026-06-13

## Quick Reference
- **Course**: Machine Learning for the Quantified Self — Vrije Universiteit Amsterdam
- **Project**: Transformer-Based Exercise Classification from Smartphone Inertial Sensors
- **Team**: Ahmad Ramadan[2772756], Taro Schenk[2781210], Jaden van Rijswijk[2873146]
- **Domain**: Human motion recognition / exercise classification from wearable sensors

## Project Identity

```
Project Name: Transformer-Based Exercise Classification
Tagline: Classifying gym exercises from smartphone inertial sensor data using Transformers
Problem: Can Transformer models outperform classical ML on exercise classification from phone sensors?
Solution: Collect IMU data (accel/gyro/linear accel) → preprocessing → compare Transformer vs classical baselines
```

## Research Questions

| # | Research Question | Approach |
|---|------------------|----------|
| 1 | How effectively can Transformer models classify human motion exercises from smartphone inertial sensor data? | Train Transformer on windowed sensor data, compare with classical ML baselines |
| 2 | Which sensor modality (accelerometer, gyroscope, or linear acceleration) carries the strongest signal for exercise classification? | Ablation study — train models with each sensor modality individually |

## Deliverables & Deadlines

| Deadline | Date | Requirements | Length |
|----------|------|-------------|--------|
| **Intermediate 1** | 07/06/2026 23:59 | Points 1-3: Research question, data collection, noise removal & missing values | ≤5 pages |
| **Intermediate 2** | 14/06/2026 23:59 | Points 4-5: Feature engineering, train/test setup, classical ML techniques | ≤5 pages |
| **Final** | 21/06/2026 23:59 | Points 1-7: Full report incl. deep learning & critical reflection | ≤14 pages (excl. refs) |

## Assignment Requirements (7 Points)

1. **RQ & Data** — Define clear research question, measurements, and target variable
2. **Data Collection & EDA** — Collect data, exploratory analysis (cf. Chapter 2)
3. **Noise & Missing Values** — Apply appropriate techniques (cf. Chapter 3)
4. **Feature Engineering** — Engineer & analyze features (cf. Chapter 4)
5. **Classical ML** — Train/test setup + classical ML techniques (cf. Chapter 7)
6. **Deep Learning** — LSTM, TCN, or Transformer with temporal embedding (state-of-the-art)
7. **Conclusion** — General conclusion and critical reflection

## Success Criteria

| Criterion | Definition |
|-----------|-----------|
| **Intermediate 1** | Pass/fail — must pass to continue |
| **Intermediate 2** | Pass/fail — must pass to continue |
| **Final Report** | Sufficient grade (combined with exam) |
| **Evaluation** | Generalization-focused (e.g., across persons or new recordings, NOT random timepoint splits) |

## Key Stakeholders

| Role | Name | Responsibility |
|------|------|---------------|
| Student | Ahmad Ramadan[2772756] | Team member |
| Student | Taro Schenk[2781210] | Team member |
| Student | Jaden van Rijswijk[2873146] | Team member |
| Instructor | Mark Hoogendoorn | Course coordinator, author of ML4QS book |
| TA | TBD | Provides weekly feedback on progress |

## 📂 Codebase References

| File | Purpose |
|------|---------|
| `Chapters/` | Reference implementations from ML4QS book (Chapters 2-8) |
| `Chapters/crowdsignals_ch*.py` | Chapter scripts with processing pipelines |
| `Chapters/util/util.py` | Utility functions (normalize, distance, statistics) |
| `Chapters/Chapter2/CreateDataset.py` | Dataset creation from sensor CSV files |
| `pyproject.toml` | Python dependencies |

## Related Files
- `technical-domain.md` — Tech stack, sensors, models, data pipeline
- `decisions-log.md` — Key decisions and rationale
- `living-notes.md` — Active issues and open questions
- `business-tech-bridge.md` — Business → technical mapping
