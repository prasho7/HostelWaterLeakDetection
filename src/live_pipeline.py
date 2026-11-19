"""
live_pipeline.py
Phase 2/3 — Automatic incremental processing of live meter readings.

Workflow per run
----------------
1. Load readings.csv; identify rows that are *newer* than the last-processed
   watermark per meter (stored in data/.watermark.json).
2. If no new rows exist, exit without touching any file.
3. Compute delta_litres for the new rows, using the last cumulative reading
   already in processed_readings.csv as the diff anchor.
4. Append the enriched new rows to processed_readings.csv (never rewrite).
5. Update the watermark file so the next run only sees genuinely new rows.
6. Re-run the detection engine:
     --rolling  → rolling_detector.run()  (Phase 3 rolling-window MAD)
     (default)  → leak_detector.run()     (Phase 2 global-dataset MAD)
7. Re-run alert_manager.main() — it already preserves existing alert IDs and
   warden-set statuses via merge_with_existing().

No existing detection logic or alert system is modified.

Usage (from project root):
    python src/live_pipeline.py                  # global MAD (default)
    python src/live_pipeline.py --rolling        # rolling-window MAD (Phase 3)
    python src/live_pipeline.py --dry-run        # report new rows, no writes
    python src/live_pipeline.py --rolling --window-days 14  # 14-day window
"""

import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
for _p in (PROJECT_ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from config import METERS, METER_IDS  # noqa: E402

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_DIR          = os.path.join(PROJECT_ROOT, "data")
READINGS_CSV      = os.path.join(DATA_DIR, "readings.csv")
PROCESSED_CSV     = os.path.join(DATA_DIR, "processed_readings.csv")
WATERMARK_FILE    = os.path.join(DATA_DIR, ".watermark.json")

# Column orders must match the existing pipeline's output exactly
READINGS_COLS    = ["timestamp", "hostel_id", "floor_id", "meter_id", "meter_reading"]
PROCESSED_COLS   = [
    "timestamp", "hostel_id", "floor_id", "meter_id",
    "meter_reading", "delta_litres", "hour", "day_of_week",
]


# ---------------------------------------------------------------------------
# Watermark helpers
# ---------------------------------------------------------------------------

def load_watermark(path: str = WATERMARK_FILE) -> dict:
    """
    Return {meter_id: last_processed_timestamp_isostring}.
    Missing file or missing meter → timestamp of Unix epoch (process everything).
    """
    epoch = "1970-01-01T00:00:00"
    marks = {m["meter_id"]: epoch for m in METERS}

    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                stored = json.load(fh)
            for mid in METER_IDS:
                if mid in stored:
                    marks[mid] = stored[mid]
        except Exception:
            pass   # corrupted watermark — reprocess everything safely

    return marks


def save_watermark(marks: dict, path: str = WATERMARK_FILE) -> None:
    """Persist watermark dict as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(marks, fh, indent=2)


# ---------------------------------------------------------------------------
# Anchor helpers (last cumulative reading already processed per meter)
# ---------------------------------------------------------------------------

def load_last_processed_readings(path: str = PROCESSED_CSV) -> dict:
    """
    Return {meter_id: last meter_reading} from processed_readings.csv.
    Falls back to 0.0 for any meter not yet represented.
    """
    anchors = {m["meter_id"]: 0.0 for m in METERS}

    if not os.path.isfile(path):
        return anchors

    try:
        df = pd.read_csv(path, usecols=["meter_id", "meter_reading"])
        for mid, grp in df.groupby("meter_id"):
            if mid in anchors:
                anchors[mid] = float(grp["meter_reading"].iloc[-1])
    except Exception:
        pass

    return anchors


# ---------------------------------------------------------------------------
# New-row detection
# ---------------------------------------------------------------------------

def find_new_rows(
    readings_path: str = READINGS_CSV,
    watermark: dict | None = None,
) -> pd.DataFrame:
    """
    Load readings.csv and return only rows with timestamps strictly newer than
    the watermark for their meter.

    Parameters
    ----------
    readings_path : path to readings.csv
    watermark     : {meter_id: last_seen_iso} — defaults to loading from disk

    Returns
    -------
    DataFrame of new rows (may be empty), sorted by meter_id then timestamp.
    Columns: timestamp (datetime64), hostel_id, floor_id, meter_id, meter_reading.
    """
    if watermark is None:
        watermark = load_watermark()

    if not os.path.isfile(readings_path):
        return pd.DataFrame(columns=READINGS_COLS)

    df = pd.read_csv(readings_path, parse_dates=["timestamp"])

    # Drop rows with invalid timestamps or meter_readings
    df = df.dropna(subset=["timestamp", "meter_reading", "meter_id"])

    # Build a per-meter cutoff Series for vectorised filtering
    cutoff_map = {
        mid: pd.Timestamp(ts) for mid, ts in watermark.items()
    }

    masks = []
    for mid, cutoff in cutoff_map.items():
        meter_mask = (df["meter_id"] == mid) & (df["timestamp"] > cutoff)
        masks.append(meter_mask)

    if not masks:
        return pd.DataFrame(columns=READINGS_COLS)

    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m

    new_df = df[combined].copy()
    new_df = new_df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)
    return new_df


# ---------------------------------------------------------------------------
# Enrichment (matches data_processor.process() logic exactly)
# ---------------------------------------------------------------------------

def enrich_new_rows(
    new_df: pd.DataFrame,
    anchors: dict,
) -> pd.DataFrame:
    """
    Add delta_litres, hour, day_of_week to new rows.

    delta_litres for the *first* new row of each meter is computed against the
    last already-processed reading (the anchor). Subsequent new rows within the
    same batch diff against each other normally.

    Parameters
    ----------
    new_df  : new rows with [timestamp, hostel_id, floor_id, meter_id, meter_reading]
    anchors : {meter_id: last_processed_meter_reading}

    Returns
    -------
    Enriched DataFrame with PROCESSED_COLS column order.
    """
    if new_df.empty:
        return pd.DataFrame(columns=PROCESSED_COLS)

    rows = []
    for mid, grp in new_df.groupby("meter_id", sort=True):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        prev = anchors.get(mid, 0.0)

        for _, row in grp.iterrows():
            current  = float(row["meter_reading"])
            delta    = max(current - prev, 0.0)   # clip negative (guard)
            prev     = current
            ts       = row["timestamp"]

            rows.append({
                "timestamp":     ts.isoformat(),
                "hostel_id":     row["hostel_id"],
                "floor_id":      row["floor_id"],
                "meter_id":      mid,
                "meter_reading": round(current, 4),
                "delta_litres":  round(delta, 4),
                "hour":          int(ts.hour),
                "day_of_week":   int(ts.day_of_week),
            })

    return pd.DataFrame(rows, columns=PROCESSED_COLS)


# ---------------------------------------------------------------------------
# Append to processed_readings.csv
# ---------------------------------------------------------------------------

def append_to_processed(enriched_df: pd.DataFrame, path: str = PROCESSED_CSV) -> None:
    """
    Append enriched rows to processed_readings.csv.
    Creates the file with header if it does not exist.
    """
    write_header = not os.path.isfile(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    enriched_df.to_csv(path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    readings_path: str = READINGS_CSV,
    processed_path: str = PROCESSED_CSV,
    watermark_path: str = WATERMARK_FILE,
    dry_run: bool = False,
    use_rolling: bool = False,
    window_days: int | None = None,
    min_readings: int | None = None,
) -> dict:
    """
    Execute one incremental pipeline cycle.

    Parameters
    ----------
    use_rolling  : if True, use rolling_detector instead of leak_detector
    window_days  : rolling window size in days (overrides config default)
    min_readings : minimum readings for rolling baseline (overrides config default)

    Returns a summary dict with keys:
        new_rows_found   : int
        new_rows_written : int
        detection_ran    : bool
        alerts_ran       : bool
        skipped          : bool   (True when no new rows)
        detector_mode    : "rolling" | "global"
    """
    summary = {
        "new_rows_found":   0,
        "new_rows_written": 0,
        "detection_ran":    False,
        "alerts_ran":       False,
        "skipped":          False,
        "detector_mode":    "rolling" if use_rolling else "global",
    }

    # 1. Identify new rows
    watermark = load_watermark(watermark_path)
    new_df    = find_new_rows(readings_path, watermark)

    summary["new_rows_found"] = len(new_df)

    if new_df.empty:
        print("[live_pipeline] No new readings detected. Nothing to do.")
        summary["skipped"] = True
        return summary

    print(f"[live_pipeline] {len(new_df)} new reading(s) found across "
          f"{new_df['meter_id'].nunique()} meter(s).")

    if dry_run:
        print("[live_pipeline] Dry-run: no files written.")
        return summary

    # 2. Compute delta_litres anchored at last processed reading
    anchors     = load_last_processed_readings(processed_path)
    enriched_df = enrich_new_rows(new_df, anchors)

    # 3. Append enriched rows to processed_readings.csv
    append_to_processed(enriched_df, processed_path)
    summary["new_rows_written"] = len(enriched_df)
    print(f"[live_pipeline] Appended {len(enriched_df)} enriched row(s) "
          f"to {processed_path}.")

    # 4. Update watermark (advance per meter to the latest timestamp just written)
    for mid, grp in new_df.groupby("meter_id"):
        watermark[mid] = grp["timestamp"].max().isoformat()
    save_watermark(watermark, watermark_path)

    # 5. Run detection engine
    if use_rolling:
        from rolling_detector import run as run_detection  # noqa: PLC0415
        kwargs = {}
        if window_days  is not None: kwargs["window_days"]  = window_days
        if min_readings is not None: kwargs["min_readings"] = min_readings
        run_detection(**kwargs)
        print("[live_pipeline] Rolling-window detection engine completed.")
    else:
        from leak_detector import run as run_detection  # noqa: PLC0415
        run_detection()
        print("[live_pipeline] Detection engine completed.")
    summary["detection_ran"] = True

    # 6. Re-run alert manager (preserves existing IDs and warden statuses)
    from alert_manager import main as run_alerts    # noqa: PLC0415
    run_alerts()
    summary["alerts_ran"] = True
    print("[live_pipeline] Alert manager completed.")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 2/3 live pipeline — incremental meter reading processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Identify new rows but do not write any files",
    )
    p.add_argument(
        "--rolling", action="store_true",
        help="Use rolling-window MAD baseline (Phase 3) instead of global MAD",
    )
    p.add_argument(
        "--window-days", type=int, default=None,
        help="Rolling window size in days (default: ROLLING_WINDOW_DAYS from config)",
    )
    p.add_argument(
        "--min-readings", type=int, default=None,
        help="Minimum readings for rolling baseline before falling back to global",
    )
    return p


def main():
    args = _build_parser().parse_args()
    run_pipeline(
        dry_run=args.dry_run,
        use_rolling=args.rolling,
        window_days=args.window_days,
        min_readings=args.min_readings,
    )


if __name__ == "__main__":
    main()
