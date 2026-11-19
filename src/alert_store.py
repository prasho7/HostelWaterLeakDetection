"""
alert_store.py — Alert Resolution Persistence Layer

Provides pure functions for reading, resolving, reopening, and saving
alert status changes to alerts.csv.  No Streamlit dependency.

The only columns this module touches on existing rows are:
  - status      : "OPEN" | "RESOLVED"
  - resolved_at : ISO 8601 datetime string | ""

All other columns (alert_id, hostel_id, floor_id, meter_id, first_seen,
last_seen, severity, avg_z_score, persistence_count) are read-only here.
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Project root — allows this module to be run as a script or imported
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Default path (can be overridden in tests)
DEFAULT_ALERTS_PATH = os.path.join(_PROJECT_ROOT, "data", "alerts.csv")

# Column order that every saved file must follow
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
    "resolved_at",
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_alerts(path: str = DEFAULT_ALERTS_PATH) -> pd.DataFrame:
    """
    Load alerts.csv and return a DataFrame with all required columns.
    If resolved_at is missing (old CSV), it is added as empty strings.
    Returns an empty DataFrame with correct columns if the file is absent.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=ALERT_COLUMNS)

    df = pd.read_csv(path, dtype=str)   # read everything as str to avoid type fights

    # Back-fill resolved_at for CSVs produced before this feature
    if "resolved_at" not in df.columns:
        df["resolved_at"] = ""

    # Normalise: ensure NaN → ""
    df["resolved_at"] = df["resolved_at"].fillna("")

    # Return only the canonical column order (extra cols silently dropped)
    return df[ALERT_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Match / Lookup
# ---------------------------------------------------------------------------

def find_matching_alert_id(
    df: pd.DataFrame,
    alert_id: str = None,
    meter_id: str = None,
    first_seen: str = None,
    last_seen: str = None,
) -> str | None:
    """
    Find a matching alert_id in df.
    First checks for an exact alert_id match.
    If not found, falls back to matching by (meter_id, first_seen, last_seen)
    to handle cases where UI has a stale alert_id across re-runs.
    Returns the matching alert_id string, or None if not found.
    """
    if df is None or df.empty:
        return None

    if alert_id:
        alert_id_str = str(alert_id).strip()
        mask = df["alert_id"].astype(str).str.strip() == alert_id_str
        if mask.any():
            return str(df.loc[mask, "alert_id"].iloc[0])

    if meter_id and first_seen and last_seen:
        meter_str = str(meter_id).strip()
        try:
            target_first = pd.to_datetime(first_seen)
            target_last  = pd.to_datetime(last_seen)
            df_first     = pd.to_datetime(df["first_seen"])
            df_last      = pd.to_datetime(df["last_seen"])

            mask = (
                (df["meter_id"].astype(str).str.strip() == meter_str)
                & (df_first == target_first)
                & (df_last == target_last)
            )
            if mask.any():
                return str(df.loc[mask, "alert_id"].iloc[0])
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------

def resolve_alert(
    df: pd.DataFrame,
    alert_id: str,
    resolved_at: str = None,
) -> pd.DataFrame:
    """
    Mark a single alert as RESOLVED.

    Parameters
    ----------
    df          : alerts DataFrame (from load_alerts)
    alert_id    : UUID string of the alert to resolve
    resolved_at : ISO 8601 timestamp string; defaults to current UTC time

    Returns
    -------
    Modified copy of df (original unchanged).

    Raises
    ------
    ValueError if alert_id is not found or is already RESOLVED.
    """
    if resolved_at is None:
        resolved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    result = df.copy()
    mask = result["alert_id"] == alert_id

    if not mask.any():
        raise ValueError(f"alert_id not found: {alert_id!r}")

    current_status = result.loc[mask, "status"].iloc[0]
    if current_status == "RESOLVED":
        raise ValueError(f"Alert {alert_id!r} is already RESOLVED.")

    result.loc[mask, "status"]      = "RESOLVED"
    result.loc[mask, "resolved_at"] = resolved_at
    return result


# ---------------------------------------------------------------------------
# Reopen
# ---------------------------------------------------------------------------

def reopen_alert(df: pd.DataFrame, alert_id: str) -> pd.DataFrame:
    """
    Reopen a RESOLVED alert (set status back to OPEN, clear resolved_at).

    Parameters
    ----------
    df       : alerts DataFrame (from load_alerts)
    alert_id : UUID string of the alert to reopen

    Returns
    -------
    Modified copy of df (original unchanged).

    Raises
    ------
    ValueError if alert_id is not found or is already OPEN.
    """
    result = df.copy()
    mask = result["alert_id"] == alert_id

    if not mask.any():
        raise ValueError(f"alert_id not found: {alert_id!r}")

    current_status = result.loc[mask, "status"].iloc[0]
    if current_status == "OPEN":
        raise ValueError(f"Alert {alert_id!r} is already OPEN.")

    result.loc[mask, "status"]      = "OPEN"
    result.loc[mask, "resolved_at"] = ""
    return result


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_alerts(df: pd.DataFrame, path: str = DEFAULT_ALERTS_PATH) -> None:
    """
    Persist the alerts DataFrame back to alerts.csv.
    Always writes the canonical ALERT_COLUMNS column order.
    Preserves all rows and existing alert_ids.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[ALERT_COLUMNS].to_csv(path, index=False)
