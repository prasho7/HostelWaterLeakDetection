"""
leak_detector.py — Hostel Water Leak Detection Engine

Reads processed_readings.csv, computes per-(meter_id, hour) modified Z-scores
using Median Absolute Deviation, flags anomalies, computes persistence counts,
and writes all rows to anomalies.csv.

Run from project root:
    python src/leak_detector.py
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so config.py is importable when running as a script
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import MAD_Z_THRESHOLD  # noqa: E402

# ---------------------------------------------------------------------------
# File paths (relative to project root)
# ---------------------------------------------------------------------------
INPUT_PATH  = os.path.join(PROJECT_ROOT, "data", "processed_readings.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "anomalies.csv")


def compute_mad_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (meter_id, hour) group compute median_baseline and mad,
    then broadcast those values back onto every row via transform.
    Returns df with two new columns: median_baseline, mad.
    """
    grouped = df.groupby(["meter_id", "hour"])["delta_litres"]

    # median_baseline: median of delta_litres for the group
    df["median_baseline"] = grouped.transform("median")

    # mad: median of |delta_litres - median_baseline| per group
    df["mad"] = grouped.transform(
        lambda x: (x - x.median()).abs().median()
    )
    return df


def compute_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add z_score_mad column.
    Formula: 0.6745 × (delta_litres − median_baseline) / max(MAD, 0.001)
    """
    denominator = df["mad"].clip(lower=0.001)
    df["z_score_mad"] = 0.6745 * (df["delta_litres"] - df["median_baseline"]) / denominator
    return df


def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_anomaly column: True when z_score_mad > MAD_Z_THRESHOLD."""
    df["is_anomaly"] = df["z_score_mad"] > MAD_Z_THRESHOLD
    return df


def compute_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each meter, walk rows sorted by timestamp and compute persistence_count:
    - increments by 1 while is_anomaly is True
    - resets to 0 when is_anomaly is False
    """
    persistence = pd.Series(0, index=df.index, dtype=int)

    for _, grp_idx in df.groupby("meter_id").groups.items():
        sorted_idx = df.loc[grp_idx].sort_values("timestamp").index
        counter = 0
        for idx in sorted_idx:
            if df.at[idx, "is_anomaly"]:
                counter += 1
            else:
                counter = 0
            persistence.at[idx] = counter

    df["persistence_count"] = persistence
    return df


def run():
    # 1. Load and prepare data
    # pandas 3.x no longer auto-infers non-ISO8601 formats via parse_dates;
    # use format='mixed' so both "2024-01-01 HH:MM:SS" and ISO-T strings work.
    df = pd.read_csv(INPUT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    # 2. Compute per-(meter_id, hour) median_baseline and MAD
    df = compute_mad_stats(df)

    # 3. Compute modified Z-scores
    df = compute_z_scores(df)

    # 4. Flag anomalies
    df = flag_anomalies(df)

    # 5. Compute persistence counts per meter
    df = compute_persistence(df)

    # 6. Write output — retain all rows, ordered columns as specified
    output_cols = [
        "timestamp", "hostel_id", "floor_id", "meter_id", "delta_litres",
        "median_baseline", "mad", "z_score_mad",
        "is_anomaly", "persistence_count",
    ]
    df[output_cols].to_csv(OUTPUT_PATH, index=False)
    print(f"anomalies.csv written -> {OUTPUT_PATH}  ({len(df)} rows)")


if __name__ == "__main__":
    run()
