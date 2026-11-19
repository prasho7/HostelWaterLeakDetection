"""
rolling_detector.py — Phase 3: Rolling Window Baseline Detection Engine
==========================================================================

Replaces the global-dataset MAD baseline from leak_detector.py with a
per-meter rolling window that considers only the most recent N days of
*prior* readings when computing each row's baseline.

Design
------
For every row R belonging to meter M at clock hour H:

  window = all prior rows of M that fall within [R.timestamp - WINDOW, R.timestamp)
           AND share the same hour-of-day bucket (±HOUR_TOLERANCE)
           AND optionally share the same day-type (weekday / weekend)

  baseline  = median(delta_litres in window)
  mad       = median(|x - baseline| for x in window)
  z_score   = 0.6745 × (R.delta_litres - baseline) / max(mad, 0.001)

The current reading R is EXCLUDED from its own baseline (req 8).

Fallback
--------
If the window contains fewer than MIN_WINDOW_READINGS valid readings, the
engine falls back to the same global (meter, hour) MAD used by leak_detector.py.
The 'baseline_source' column records which path was taken ('rolling' or 'global').

Output schema
-------------
The output CSV is identical to anomalies.csv from leak_detector.run() so that
alert_manager.py and the dashboard can consume it without modification:

    timestamp, hostel_id, floor_id, meter_id, delta_litres,
    median_baseline, mad, z_score_mad, is_anomaly, persistence_count

Usage (from project root):
    python src/rolling_detector.py                      # default window
    python src/rolling_detector.py --window-days 14     # 14-day window
    python src/rolling_detector.py --min-readings 5     # lower fallback threshold
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (  # noqa: E402
    MAD_Z_THRESHOLD,
    PERSISTENCE_MIN,
    ROLLING_WINDOW_DAYS,
    ROLLING_MIN_READINGS,
)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
INPUT_PATH  = os.path.join(PROJECT_ROOT, "data", "processed_readings.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "anomalies.csv")

# ---------------------------------------------------------------------------
# Rolling window defaults  (config.py values used; can be overridden at call)
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_DAYS    = ROLLING_WINDOW_DAYS    # look back this many days
DEFAULT_MIN_READINGS   = ROLLING_MIN_READINGS   # minimum prior readings required
HOUR_TOLERANCE         = 1                      # ±1 hour around target hour


# ---------------------------------------------------------------------------
# Core baseline computation
# ---------------------------------------------------------------------------

def _compute_global_stats(df: pd.DataFrame) -> dict:
    """
    Pre-compute global per-(meter_id, hour) median + MAD as the fallback
    baseline.  Returns a dict keyed by (meter_id, hour) → (median, mad).
    Uses ALL rows in df (the same logic as leak_detector.compute_mad_stats).
    """
    stats = {}
    for (mid, hr), grp in df.groupby(["meter_id", "hour"]):
        vals = grp["delta_litres"].values
        med  = float(np.median(vals))
        mad  = float(np.median(np.abs(vals - med)))
        stats[(mid, hr)] = (med, mad)
    return stats


def compute_rolling_baselines(
    df: pd.DataFrame,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_readings: int = DEFAULT_MIN_READINGS,
    hour_tolerance: int = HOUR_TOLERANCE,
) -> pd.DataFrame:
    """
    For every row in df, compute a rolling-window median baseline and MAD
    using only prior readings from the same meter within the look-back window.

    Parameters
    ----------
    df            : processed_readings DataFrame, sorted by (meter_id, timestamp)
    window_days   : number of days to look back for the baseline window
    min_readings  : minimum prior readings in window to use rolling baseline;
                    falls back to global stats when fewer are available
    hour_tolerance: include readings ±hour_tolerance hours around the target hour

    Returns
    -------
    df with added columns:
        median_baseline, mad, z_score_mad, is_anomaly, baseline_source
    """
    window_td    = pd.Timedelta(days=window_days)
    global_stats = _compute_global_stats(df)

    median_baselines = np.empty(len(df), dtype=float)
    mads             = np.empty(len(df), dtype=float)
    sources          = np.empty(len(df), dtype=object)

    # Process each meter independently — keeps per-meter isolation
    for mid, meter_df in df.groupby("meter_id", sort=False):
        meter_df = meter_df.sort_values("timestamp")
        idxs     = meter_df.index.tolist()
        deltas   = meter_df["delta_litres"].values
        tss      = meter_df["timestamp"].values        # numpy datetime64
        hours    = meter_df["hour"].values

        for pos, idx in enumerate(idxs):
            ts_cur   = tss[pos]
            hr_cur   = int(hours[pos])
            ts_start = ts_cur - window_td.to_timedelta64()

            # Collect prior readings within window that share similar hour-of-day
            # Requirement 8: strictly BEFORE current row (pos not included)
            prior_mask = (
                (tss[:pos] >= ts_start) &
                (tss[:pos] < ts_cur) &
                (np.abs(hours[:pos].astype(int) - hr_cur) <= hour_tolerance)
            )
            prior_deltas = deltas[:pos][prior_mask]

            if len(prior_deltas) >= min_readings:
                med = float(np.median(prior_deltas))
                mad = float(np.median(np.abs(prior_deltas - med)))
                src = "rolling"
            else:
                # Fallback: global (meter_id, hour) baseline
                med, mad = global_stats.get((mid, hr_cur), (0.0, 0.0))
                src = "global"

            median_baselines[idx] = med
            mads[idx]             = mad
            sources[idx]          = src

    df = df.copy()
    df["median_baseline"] = median_baselines
    df["mad"]             = mads
    df["baseline_source"] = sources

    # Modified Z-score
    denominator   = pd.Series(mads, index=df.index).clip(lower=0.001)
    df["z_score_mad"] = 0.6745 * (df["delta_litres"] - df["median_baseline"]) / denominator

    # Anomaly flag
    df["is_anomaly"] = df["z_score_mad"] > MAD_Z_THRESHOLD

    return df


# ---------------------------------------------------------------------------
# Persistence (identical logic to leak_detector.compute_persistence)
# ---------------------------------------------------------------------------

def compute_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-meter streak counter: increment while is_anomaly True, reset on False.
    Identical algorithm to leak_detector.compute_persistence — kept here so
    rolling_detector is self-contained and testable independently.
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


# ---------------------------------------------------------------------------
# Output column order (must match anomalies.csv schema exactly)
# ---------------------------------------------------------------------------

OUTPUT_COLS = [
    "timestamp", "hostel_id", "floor_id", "meter_id", "delta_litres",
    "median_baseline", "mad", "z_score_mad",
    "is_anomaly", "persistence_count",
]


# ---------------------------------------------------------------------------
# Main entry point (mirrors leak_detector.run() signature)
# ---------------------------------------------------------------------------

def run(
    input_path:   str = INPUT_PATH,
    output_path:  str = OUTPUT_PATH,
    window_days:  int = DEFAULT_WINDOW_DAYS,
    min_readings: int = DEFAULT_MIN_READINGS,
) -> pd.DataFrame:
    """
    Load processed_readings.csv, compute rolling baselines, flag anomalies,
    compute persistence, and write anomalies.csv.

    Returns the full anomalies DataFrame (useful for testing).
    """
    # Load — handle mixed ISO / space-separated timestamp formats (pandas 3.x)
    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    # Rolling baseline + anomaly flag
    df = compute_rolling_baselines(df, window_days=window_days, min_readings=min_readings)

    # Persistence counts
    df = compute_persistence(df)

    # Write output — same schema as anomalies.csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[OUTPUT_COLS].to_csv(output_path, index=False)
    print(
        f"[rolling_detector] anomalies.csv written -> {output_path}  "
        f"({len(df)} rows, window={window_days}d, min_readings={min_readings})"
    )

    return df[OUTPUT_COLS]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 3 rolling-window leak detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
        help="Days of history to include in the rolling baseline window",
    )
    p.add_argument(
        "--min-readings", type=int, default=DEFAULT_MIN_READINGS,
        help="Minimum prior readings needed before using rolling (else global fallback)",
    )
    p.add_argument(
        "--input",  default=INPUT_PATH,  help="Path to processed_readings.csv"
    )
    p.add_argument(
        "--output", default=OUTPUT_PATH, help="Path for anomalies.csv output"
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        window_days=args.window_days,
        min_readings=args.min_readings,
    )
