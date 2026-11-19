"""
live_simulator.py
Phase 1 — Live monitoring simulator for the Hostel Water Leak Detector.

Generates new meter readings for all 12 floor-level meters across 3 hostels
and appends them to data/readings.csv without touching historical data.

Scenario types (per reading tick):
  - normal              : typical daytime/night consumption
  - spike               : sudden single-tick surge (e.g. bulk flush, burst)
  - continuous_leak     : sustained excess flow over multiple ticks
  - abnormal_night      : elevated consumption during night hours

Usage (run from project root):
    python src/live_simulator.py                       # 1 batch, default interval
    python src/live_simulator.py --ticks 5             # 5 successive batches
    python src/live_simulator.py --interval 30         # 30-minute tick interval
    python src/live_simulator.py --ticks 3 --dry-run   # preview without writing
"""

import argparse
import os
import sys
import random
import math
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from config import METERS, METER_IDS

# ---------------------------------------------------------------------------
# Simulation constants  (all in litres)
# ---------------------------------------------------------------------------

# Normal daytime consumption per hour at floor level (~4 rooms)
NORMAL_MEAN = 60.0
NORMAL_STD  = 10.0
NORMAL_MIN  = 8.0

# Night-hour band (23:00 – 05:00 inclusive)
NIGHT_HOURS      = set(range(23, 24)) | set(range(0, 6))
NIGHT_MULTIPLIER = 0.3

# Spike scenario: one-off surge on top of normal
SPIKE_EXTRA_MIN = 200.0
SPIKE_EXTRA_MAX = 500.0

# Continuous-leak scenario: persistent excess per tick
LEAK_EXTRA_MIN = 150.0
LEAK_EXTRA_MAX = 300.0

# Abnormal-night scenario: night consumption at this fraction of daytime instead
ABNORMAL_NIGHT_MULTIPLIER = 0.9   # close to daytime — clearly anomalous

# Default tick interval (minutes) — overridable at CLI or call level
DEFAULT_INTERVAL_MINS = 60

# Scenario weights used when picking randomly (must sum to 1.0)
SCENARIO_WEIGHTS = {
    "normal":           0.80,
    "spike":            0.05,
    "continuous_leak":  0.10,
    "abnormal_night":   0.05,
}

# ---------------------------------------------------------------------------
# CSV path
# ---------------------------------------------------------------------------
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
READINGS_CSV = os.path.join(DATA_DIR, "readings.csv")

COLUMN_ORDER = ["timestamp", "hostel_id", "floor_id", "meter_id", "meter_reading"]


# ---------------------------------------------------------------------------
# State management  (last cumulative reading per meter)
# ---------------------------------------------------------------------------

def load_last_readings(csv_path: str = READINGS_CSV) -> dict:
    """
    Return the last cumulative meter_reading per meter_id from an existing CSV.
    Falls back to 0.0 for any meter not yet present in the file.
    """
    last = {m["meter_id"]: 0.0 for m in METERS}

    if not os.path.isfile(csv_path):
        return last

    # Read only the columns we need; avoid loading full history into memory
    try:
        df = pd.read_csv(csv_path, usecols=["meter_id", "meter_reading"])
        for meter_id, group in df.groupby("meter_id"):
            if meter_id in last:
                last[meter_id] = float(group["meter_reading"].iloc[-1])
    except Exception:
        pass   # corrupted or empty file — start from 0

    return last


def load_last_timestamps(csv_path: str = READINGS_CSV) -> dict:
    """
    Return the last timestamp per meter_id from an existing CSV.
    Falls back to current UTC time for any meter not found.
    """
    now = datetime.utcnow().replace(second=0, microsecond=0)
    last = {m["meter_id"]: now for m in METERS}

    if not os.path.isfile(csv_path):
        return last

    try:
        df = pd.read_csv(csv_path, usecols=["meter_id", "timestamp"])
        for meter_id, group in df.groupby("meter_id"):
            if meter_id in last:
                ts_str = group["timestamp"].iloc[-1]
                last[meter_id] = datetime.fromisoformat(str(ts_str))
    except Exception:
        pass

    return last


# ---------------------------------------------------------------------------
# Consumption engine
# ---------------------------------------------------------------------------

def _normal_delta(hour: int, rng: random.Random) -> float:
    """Return a base consumption value for one tick (normal scenario)."""
    delta = rng.gauss(NORMAL_MEAN, NORMAL_STD)
    delta = max(delta, NORMAL_MIN)
    if hour in NIGHT_HOURS:
        delta *= NIGHT_MULTIPLIER
    return delta


def compute_delta(scenario: str, hour: int, rng: random.Random) -> float:
    """
    Return litres consumed for one interval tick under the given scenario.

    Parameters
    ----------
    scenario : one of "normal", "spike", "continuous_leak", "abnormal_night"
    hour     : 0–23 hour of the tick timestamp
    rng      : random.Random instance
    """
    if scenario == "normal":
        return _normal_delta(hour, rng)

    if scenario == "spike":
        base  = _normal_delta(hour, rng)
        extra = rng.uniform(SPIKE_EXTRA_MIN, SPIKE_EXTRA_MAX)
        return base + extra

    if scenario == "continuous_leak":
        base  = _normal_delta(hour, rng)
        extra = rng.uniform(LEAK_EXTRA_MIN, LEAK_EXTRA_MAX)
        return base + extra

    if scenario == "abnormal_night":
        # Night hours but consuming at near-daytime rate
        base = rng.gauss(NORMAL_MEAN, NORMAL_STD)
        base = max(base, NORMAL_MIN)
        if hour in NIGHT_HOURS:
            base *= ABNORMAL_NIGHT_MULTIPLIER   # stays high instead of dropping
        return base

    raise ValueError(f"Unknown scenario: {scenario!r}")


def _pick_scenario(rng: random.Random) -> str:
    """Randomly pick a scenario according to SCENARIO_WEIGHTS."""
    r = rng.random()
    cumulative = 0.0
    for scenario, weight in SCENARIO_WEIGHTS.items():
        cumulative += weight
        if r < cumulative:
            return scenario
    return "normal"


# ---------------------------------------------------------------------------
# Single-tick batch generator
# ---------------------------------------------------------------------------

def generate_tick(
    last_readings: dict,
    last_timestamps: dict,
    interval_mins: int = DEFAULT_INTERVAL_MINS,
    scenario_overrides: dict | None = None,
    rng: random.Random | None = None,
) -> pd.DataFrame:
    """
    Generate one reading per meter for the next tick and return as a DataFrame.

    Parameters
    ----------
    last_readings      : {meter_id: cumulative_reading} from previous state
    last_timestamps    : {meter_id: datetime} of previous reading per meter
    interval_mins      : minutes between readings
    scenario_overrides : optional {meter_id: scenario} to force a scenario
    rng                : optional seeded random.Random for reproducibility

    Returns
    -------
    DataFrame with columns: timestamp, hostel_id, floor_id, meter_id, meter_reading
    """
    if rng is None:
        rng = random.Random()

    rows = []
    for meter in METERS:
        hostel_id = meter["hostel_id"]
        floor_id  = meter["floor_id"]
        meter_id  = meter["meter_id"]

        prev_ts      = last_timestamps[meter_id]
        next_ts      = prev_ts + timedelta(minutes=interval_mins)
        prev_reading = last_readings[meter_id]

        scenario = (scenario_overrides or {}).get(meter_id) or _pick_scenario(rng)
        delta    = compute_delta(scenario, next_ts.hour, rng)

        new_reading = round(prev_reading + delta, 4)

        rows.append({
            "timestamp":     next_ts.isoformat(),
            "hostel_id":     hostel_id,
            "floor_id":      floor_id,
            "meter_id":      meter_id,
            "meter_reading": new_reading,
        })

    return pd.DataFrame(rows, columns=COLUMN_ORDER)


# ---------------------------------------------------------------------------
# Multi-tick runner
# ---------------------------------------------------------------------------

def run_simulation(
    ticks: int = 1,
    interval_mins: int = DEFAULT_INTERVAL_MINS,
    csv_path: str = READINGS_CSV,
    dry_run: bool = False,
    rng: random.Random | None = None,
) -> pd.DataFrame:
    """
    Run `ticks` successive reading generations and append them to the CSV.

    Parameters
    ----------
    ticks         : number of reading batches to generate
    interval_mins : minutes between consecutive readings
    csv_path      : path to the readings CSV (append target)
    dry_run       : if True, generate but do not write to disk
    rng           : optional seeded random.Random for reproducibility

    Returns
    -------
    DataFrame of all newly generated rows (ticks × 12 rows)
    """
    if rng is None:
        rng = random.Random()

    last_readings   = load_last_readings(csv_path)
    last_timestamps = load_last_timestamps(csv_path)

    all_new_rows = []

    for _ in range(ticks):
        tick_df = generate_tick(last_readings, last_timestamps, interval_mins, rng=rng)

        # Advance state for the next iteration
        for _, row in tick_df.iterrows():
            last_readings[row["meter_id"]]   = row["meter_reading"]
            last_timestamps[row["meter_id"]] = datetime.fromisoformat(row["timestamp"])

        all_new_rows.append(tick_df)

    new_df = pd.concat(all_new_rows, ignore_index=True)

    if not dry_run:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # Append without header if file already exists
        write_header = not os.path.isfile(csv_path)
        new_df.to_csv(csv_path, mode="a", header=write_header, index=False)

    return new_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live water meter reading simulator (Phase 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticks",    type=int,  default=1,
                   help="Number of reading batches to generate")
    p.add_argument("--interval", type=int,  default=DEFAULT_INTERVAL_MINS,
                   help="Minutes between readings")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print generated rows without writing to disk")
    p.add_argument("--seed",     type=int,  default=None,
                   help="Random seed for reproducibility")
    return p


def main():
    args = _build_arg_parser().parse_args()
    rng  = random.Random(args.seed) if args.seed is not None else random.Random()

    new_df = run_simulation(
        ticks=args.ticks,
        interval_mins=args.interval,
        dry_run=args.dry_run,
        rng=rng,
    )

    action = "Generated (dry-run)" if args.dry_run else "Appended"
    print(f"[live_simulator] {action} {len(new_df)} rows "
          f"({args.ticks} tick(s) x 12 meters) "
          f"with {args.interval}-min interval.")

    if args.dry_run:
        print(new_df.to_string(index=False))


if __name__ == "__main__":
    main()
