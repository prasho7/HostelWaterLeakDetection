"""
alert_manager.py — Alert Manager
Reads anomalies.csv, groups consecutive anomaly streaks per meter,
filters false alarms (streak < PERSISTENCE_MIN), and writes alerts.csv.

Run from project root:
    python src/alert_manager.py
"""

import os
import sys
import uuid

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so config.py is importable when run from any cwd.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PERSISTENCE_MIN, SEVERITY_LOW_MAX_Z, SEVERITY_MEDIUM_MAX_Z  # noqa: E402

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
ANOMALIES_PATH = os.path.join(PROJECT_ROOT, "data", "anomalies.csv")
ALERTS_PATH    = os.path.join(PROJECT_ROOT, "data", "alerts.csv")

# Output column order
ALERT_COLUMNS = [
    "alert_id",
    "hostel_id",
    "floor_id",
    "meter_id",
    "first_seen",
    "last_seen",
    "severity",
    "avg_z_score",
    "persistence_count",
    "status",
    "resolved_at",   # ISO datetime string when resolved; empty string when OPEN
]


def classify_severity(avg_z: float) -> str:
    """Return severity label based on average modified Z-score."""
    if avg_z <= SEVERITY_LOW_MAX_Z:
        return "LOW"
    elif avg_z <= SEVERITY_MEDIUM_MAX_Z:
        return "MEDIUM"
    else:
        return "HIGH"


def build_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a fully-loaded anomalies DataFrame (sorted by meter_id, timestamp),
    identify qualifying anomaly streaks and return an alerts DataFrame.
    """
    alerts = []

    # Process each meter independently
    for meter_id, meter_df in df.groupby("meter_id", sort=True):
        meter_df = meter_df.sort_values("timestamp").reset_index(drop=True)

        # hostel_id and floor_id are constant per meter — grab once
        hostel_id = meter_df["hostel_id"].iloc[0]
        floor_id  = meter_df["floor_id"].iloc[0]

        streak_rows = []  # accumulates rows belonging to the current streak

        for _, row in meter_df.iterrows():
            if row["is_anomaly"]:
                streak_rows.append(row)
            else:
                # Non-anomaly row breaks the streak — flush it
                if len(streak_rows) >= PERSISTENCE_MIN:
                    alerts.append(_make_alert(hostel_id, floor_id, meter_id, streak_rows))
                streak_rows = []

        # Flush any trailing streak at end of meter data
        if len(streak_rows) >= PERSISTENCE_MIN:
            alerts.append(_make_alert(hostel_id, floor_id, meter_id, streak_rows))

    if not alerts:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    return pd.DataFrame(alerts, columns=ALERT_COLUMNS)


def _make_alert(hostel_id: str, floor_id: str, meter_id: str, streak_rows: list) -> dict:
    """Build a single alert dict from a list of streak rows (pandas Series)."""
    z_scores = [r["z_score_mad"] for r in streak_rows]
    avg_z    = float(np.mean(z_scores))

    return {
        "alert_id":          str(uuid.uuid4()),
        "hostel_id":         hostel_id,
        "floor_id":          floor_id,
        "meter_id":          meter_id,
        "first_seen":        streak_rows[0]["timestamp"].isoformat(),
        "last_seen":         streak_rows[-1]["timestamp"].isoformat(),
        "severity":          classify_severity(avg_z),
        "avg_z_score":       round(avg_z, 6),
        "persistence_count": len(streak_rows),
        "status":            "OPEN",
        "resolved_at":       "",   # empty until warden resolves
    }


def merge_with_existing(new_df: pd.DataFrame, existing_path: str) -> pd.DataFrame:
    """
    Match newly-detected alerts against an existing alerts.csv so that:
      - Alerts already on disk keep their alert_id, status, and resolved_at.
      - Alerts that no longer exist in the new detection are dropped.
      - Genuinely new alerts (no match in existing) get fresh UUIDs.

    Matching key: (meter_id, first_seen, last_seen) — these are deterministic
    for a given dataset and threshold configuration.

    Parameters
    ----------
    new_df        : freshly built alerts DataFrame (all OPEN, new UUIDs)
    existing_path : path to the on-disk alerts.csv (may not exist)

    Returns
    -------
    Merged DataFrame with stable IDs for existing alerts and fresh IDs for new ones.
    """
    if not os.path.exists(existing_path):
        return new_df   # nothing to merge — first run

    existing = pd.read_csv(existing_path, dtype=str)
    if existing.empty:
        return new_df

    # Normalise resolved_at
    if "resolved_at" not in existing.columns:
        existing["resolved_at"] = ""
    existing["resolved_at"] = existing["resolved_at"].fillna("")

    # Build a lookup: (meter_id, first_seen, last_seen) → (alert_id, status, resolved_at)
    existing["_key"] = (
        existing["meter_id"] + "|" +
        existing["first_seen"].str.strip() + "|" +
        existing["last_seen"].str.strip()
    )
    lookup = existing.set_index("_key")[["alert_id", "status", "resolved_at"]].to_dict("index")

    result = new_df.copy()

    # Normalise new_df timestamps to plain ISO strings for key construction
    # (new_df timestamps are already ISO strings from _make_alert)
    result["_key"] = (
        result["meter_id"] + "|" +
        result["first_seen"].str.strip() + "|" +
        result["last_seen"].str.strip()
    )

    for idx, row in result.iterrows():
        existing_record = lookup.get(row["_key"])
        if existing_record:
            # Preserve stable ID and warden-set status/resolved_at
            result.at[idx, "alert_id"]    = existing_record["alert_id"]
            result.at[idx, "status"]      = existing_record["status"]
            result.at[idx, "resolved_at"] = existing_record["resolved_at"]
        # else: brand-new alert streak — keep the fresh UUID already in result

    result = result.drop(columns=["_key"])
    return result


def get_newly_created_alerts(merged_df: pd.DataFrame, existing_path: str = ALERTS_PATH) -> pd.DataFrame:
    """
    Identify which alerts in merged_df are newly created in this run.
    Alerts whose alert_id existed on disk prior to this run are excluded.
    """
    if merged_df.empty:
        return pd.DataFrame(columns=merged_df.columns)
    if not os.path.exists(existing_path):
        return merged_df.copy()

    try:
        existing = pd.read_csv(existing_path, dtype=str)
        if existing.empty or "alert_id" not in existing.columns:
            return merged_df.copy()
        existing_ids = set(existing["alert_id"].dropna().astype(str).str.strip())
        new_mask = ~merged_df["alert_id"].astype(str).str.strip().isin(existing_ids)
        return merged_df[new_mask].copy()
    except Exception:
        return pd.DataFrame(columns=merged_df.columns)


def main():
    print(f"[alert_manager] Reading {ANOMALIES_PATH}")
    df = pd.read_csv(ANOMALIES_PATH)

    # pandas 3.x no longer auto-infers non-ISO8601 date formats via parse_dates;
    # explicitly convert using 'mixed' so both "2024-01-01 HH:MM:SS" and
    # "2024-01-01THH:MM:SS" are handled correctly.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    # Ensure boolean type (CSV may store 'True'/'False' strings)
    df["is_anomaly"] = df["is_anomaly"].astype(str).str.strip().str.lower() == "true"

    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    alerts_df = build_alerts(df)
    print(f"[alert_manager] Generated {len(alerts_df)} alert(s).")

    # Preserve existing alert IDs and warden-set statuses
    alerts_df = merge_with_existing(alerts_df, ALERTS_PATH)
    print(f"[alert_manager] Merged with existing alerts (IDs and statuses preserved).")

    # Identify brand new alerts before writing over ALERTS_PATH
    new_alerts = get_newly_created_alerts(alerts_df, ALERTS_PATH)

    os.makedirs(os.path.dirname(ALERTS_PATH), exist_ok=True)
    alerts_df.to_csv(ALERTS_PATH, index=False)
    print(f"[alert_manager] Alerts written to {ALERTS_PATH}")

    # Dispatch SMS notifications for newly created HIGH-severity alerts
    if not new_alerts.empty:
        try:
            from sms_notifier import notify_new_alerts
            notify_new_alerts(new_alerts)
        except Exception as exc:
            print(f"[alert_manager] SMS notification skipped/failed: {exc}")


if __name__ == "__main__":
    main()
