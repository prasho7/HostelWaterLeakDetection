"""
data_generator.py
Generates synthetic floor-level water meter readings for the Hostel Water Leak Detector.
One meter per floor, 3 hostels × 4 floors = 12 meters total.
Writes output to data/readings.csv.

Run from the project root:
    python src/data_generator.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Allow running from project root: resolve config.py via its own directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    METERS,
    START_DATE,
    END_DATE,
    READING_INTERVAL_MINS,
    LEAK_INJECT_METERS,
)

# Fixed seed for full reproducibility
RANDOM_SEED = 42

# Normal floor-level consumption parameters (floor serves ~4 rooms)
CONSUMPTION_MEAN = 60.0   # litres per hour (floor-level, ~4 rooms × 15 L)
CONSUMPTION_STD  = 10.0
CONSUMPTION_MIN  = 8.0    # minimum clip

# Night-hour multiplier (23:00 – 05:00 inclusive)
NIGHT_HOURS      = set(range(23, 24)) | set(range(0, 6))   # {0,1,2,3,4,5,23}
NIGHT_MULTIPLIER = 0.3

# Leak injection parameters
LEAK_WINDOWS_PER_METER = 2       # number of random leak windows per leak meter
LEAK_MIN_HOURS         = 12
LEAK_MAX_HOURS         = 48
LEAK_EXTRA_MIN         = 150.0   # extra litres/hour during a leak (floor-level)
LEAK_EXTRA_MAX         = 300.0


def build_timestamps(start: str, end: str, interval_mins: int) -> pd.DatetimeIndex:
    """Return hourly timestamps from START_DATE 00:00 to END_DATE 23:00 (inclusive)."""
    start_dt = datetime.fromisoformat(start)
    end_dt   = datetime.fromisoformat(end) + timedelta(hours=23)
    return pd.date_range(start=start_dt, end=end_dt, freq=f"{interval_mins}min")


def pick_leak_windows(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> set:
    """
    For one leak meter, randomly choose LEAK_WINDOWS_PER_METER non-overlapping
    windows of 12–48 hours. Returns a set of integer positions inside a leak window.
    """
    n = len(timestamps)
    leak_positions = set()

    for _ in range(LEAK_WINDOWS_PER_METER):
        window_len = int(rng.integers(LEAK_MIN_HOURS, LEAK_MAX_HOURS + 1))
        max_start = n - window_len
        if max_start <= 0:
            continue
        start_idx = int(rng.integers(0, max_start))
        for i in range(start_idx, start_idx + window_len):
            leak_positions.add(i)

    return leak_positions


def generate_readings() -> pd.DataFrame:
    """Generate and return the full readings DataFrame for all 12 floor meters."""
    rng = np.random.default_rng(RANDOM_SEED)
    timestamps = build_timestamps(START_DATE, END_DATE, READING_INTERVAL_MINS)

    rows = []

    for meter in METERS:
        hostel_id = meter["hostel_id"]
        floor_id  = meter["floor_id"]
        meter_id  = meter["meter_id"]

        # Pre-compute leak window positions for this meter (empty set if no leak)
        leak_positions = set()
        if meter_id in LEAK_INJECT_METERS:
            leak_positions = pick_leak_windows(timestamps, rng)

        # Base consumption for every hour (Normal, clipped to minimum)
        base = rng.normal(loc=CONSUMPTION_MEAN, scale=CONSUMPTION_STD, size=len(timestamps))
        base = np.clip(base, CONSUMPTION_MIN, None)

        cumulative = 0.0
        for idx, ts in enumerate(timestamps):
            delta = base[idx]

            # Night-hour reduction
            if ts.hour in NIGHT_HOURS:
                delta *= NIGHT_MULTIPLIER

            # Leak injection
            if idx in leak_positions:
                delta += rng.uniform(LEAK_EXTRA_MIN, LEAK_EXTRA_MAX)

            cumulative += delta
            rows.append({
                "timestamp":     ts.isoformat(),
                "hostel_id":     hostel_id,
                "floor_id":      floor_id,
                "meter_id":      meter_id,
                "meter_reading": round(cumulative, 4),
            })

    return pd.DataFrame(rows, columns=["timestamp", "hostel_id", "floor_id", "meter_id", "meter_reading"])


def main():
    df = generate_readings()

    output_dir  = os.path.join(PROJECT_ROOT, "data")
    output_path = os.path.join(output_dir, "readings.csv")

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[data_generator] Wrote {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
