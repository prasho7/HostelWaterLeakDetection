# Hostel Water Leak Detector

A Python-based system that detects hidden hostel water leaks using synthetic cumulative meter readings, Pandas, and statistical anomaly detection (Median + MAD).

---

## Architecture

```
data_generator.py  -->  readings.csv
                              |
                     data_processor.py  -->  processed_readings.csv
                                                     |
                                           leak_detector.py  -->  anomalies.csv
                                                                         |
                                                              alert_manager.py  -->  alerts.csv
                                                                                           |
                                                                                    dashboard.py  (Streamlit)
```

All inter-module communication is via CSV files. No module imports another `src/` module.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
# From the hostel_water_leak_detector/ directory:
py src/data_generator.py
py src/data_processor.py
py src/leak_detector.py
py src/alert_manager.py
```

### 3. Launch the dashboard
```bash
streamlit run src/dashboard.py
```

### 4. Run all tests
```bash
py -m pytest tests/ -v
```

---

## Project Structure

```
hostel_water_leak_detector/
├── config.py                     # Shared constants (thresholds, room list)
├── requirements.txt
├── README.md
├── data/
│   ├── readings.csv              # Raw synthetic cumulative meter readings
│   ├── processed_readings.csv    # Cleaned readings with deltas, hour, day_of_week
│   ├── anomalies.csv             # Per-reading anomaly scores and flags
│   └── alerts.csv                # Filtered, deduplicated alerts with severity
├── src/
│   ├── data_generator.py         # Phase 1: synthetic data with injected leaks
│   ├── data_processor.py         # Phase 2: compute deltas and time features
│   ├── leak_detector.py          # Phase 3: Median/MAD anomaly detection
│   ├── alert_manager.py          # Phase 4: persistence filtering + severity
│   └── dashboard.py              # Phase 5: Streamlit warden dashboard
└── tests/
    ├── test_data_generator.py
    ├── test_data_processor.py
    ├── test_leak_detector.py
    ├── test_alert_manager.py
    └── test_dashboard.py
```

---

## Detection Algorithm

1. **Delta computation** — convert cumulative meter readings to per-hour consumption (`delta_litres`).
2. **Baseline per (room, hour)** — compute median and MAD of historical consumption for each room × hour-of-day group.
3. **Modified Z-score** — `z = 0.6745 × (delta − median) / max(MAD, 0.001)`
4. **Anomaly flag** — `is_anomaly = z > MAD_Z_THRESHOLD` (default 3.5).
5. **Persistence filter** — only raise an alert when ≥ `PERSISTENCE_MIN` (default 3) consecutive anomalies occur for the same room, eliminating one-off noise.
6. **Severity** — based on average Z-score of the streak: LOW / MEDIUM / HIGH.

---

## Synthetic Leak Injection

Rooms `R102`, `R203`, `R305` have synthetic leaks injected:
- 2 random time windows of 12–48 hours each per room.
- Extra 40–80 litres/hour added on top of normal consumption during windows.
- Fixed seed 42 ensures reproducibility.

---

## Configuration (`config.py`)

| Constant | Default | Description |
|---|---|---|
| `ROOM_IDS` | 15 rooms | List of room identifiers |
| `START_DATE` | 2024-01-01 | Simulation start |
| `END_DATE` | 2024-01-31 | Simulation end (31 days) |
| `READING_INTERVAL_MINS` | 60 | One reading per hour |
| `LEAK_INJECT_ROOMS` | R102, R203, R305 | Rooms with synthetic leaks |
| `MAD_Z_THRESHOLD` | 3.5 | Modified Z-score cutoff |
| `PERSISTENCE_MIN` | 3 | Min consecutive anomalies for alert |
| `SEVERITY_LOW_MAX_Z` | 5.0 | Z ≤ 5 → LOW |
| `SEVERITY_MEDIUM_MAX_Z` | 10.0 | Z ≤ 10 → MEDIUM; above → HIGH |
